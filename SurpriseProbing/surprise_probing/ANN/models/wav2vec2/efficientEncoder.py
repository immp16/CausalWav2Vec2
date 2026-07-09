## We observed that there is one
# very unefficient operation in the encoder as defined by huggingface,
# Where the attention_mask is repeated to set the inputs to 0
# This operation is not useful as in that case the padded inputs are set to 0
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    BaseModelOutput,
    Wav2Vec2EncoderLayer,
    Wav2Vec2EncoderLayerStableLayerNorm,
    is_deepspeed_zero3_enabled,
)


class CausalWav2Vec2PositionalConvEmbedding(nn.Module):
    """
    Positional convolution that is causal (left padding only), so positions never see future inputs.
    """

    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=0,
            groups=config.num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.left_padding = config.num_conv_pos_embeddings - 1
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        if self.left_padding > 0:
            hidden_states = F.pad(hidden_states, (self.left_padding, 0))

        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


def _build_causal_bias(
    seq_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    # Create mask with 0 on and below the diagonal, -inf above (disallows attending to the future)
    causal_mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    future_mask = torch.triu(torch.ones_like(causal_mask, dtype=torch.bool), diagonal=1)
    causal_mask = causal_mask.masked_fill(future_mask, torch.finfo(dtype).min)
    return causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)


class fastWav2Vec2CausalEncoder(nn.Module):
    """
    Encoder that enforces causal attention in the transformer stack.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = CausalWav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # Build causal bias
        seq_len = hidden_states.shape[1]
        causal_bias = _build_causal_bias(
            seq_len, hidden_states.dtype, hidden_states.device
        )

        additive_attention = causal_bias

        if attention_mask is not None:
            if len(attention_mask.shape) == 3:
                # latent mask provided; convert to additive and combine with causal bias
                am = 1.0 - attention_mask[:, None, :, :].to(dtype=hidden_states.dtype)
                am = am * torch.finfo(hidden_states.dtype).min
                additive_attention = additive_attention + am
            else:
                # padding mask: zero out padding tokens and create additive bias
                hidden_states = attention_mask[..., None] * hidden_states
                am = 1.0 - attention_mask[:, None, None, :].to(
                    dtype=hidden_states.dtype
                )
                am = am * torch.finfo(hidden_states.dtype).min
                additive_attention = additive_attention + am

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            dropout_probability = np.random.uniform(0, 1)
            skip_the_layer = (
                True
                if self.training and (dropout_probability < self.config.layerdrop)
                else False
            )
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        additive_attention,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=additive_attention,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class fastWav2Vec2CausalEncoderStableLayerNorm(nn.Module):
    """
    Causal encoder that uses stable LayerNorm encoder layers.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = CausalWav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [
                Wav2Vec2EncoderLayerStableLayerNorm(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        seq_len = hidden_states.shape[1]
        additive_attention = _build_causal_bias(
            seq_len, hidden_states.dtype, hidden_states.device
        )

        if attention_mask is not None:
            if len(attention_mask.shape) == 3:
                am = 1.0 - attention_mask[:, None, :, :].to(dtype=hidden_states.dtype)
                am = am * torch.finfo(hidden_states.dtype).min
                additive_attention = additive_attention + am
            else:
                hidden_states = attention_mask[..., None] * hidden_states
                am = 1.0 - attention_mask[:, None, None, :].to(
                    dtype=hidden_states.dtype
                )
                am = am * torch.finfo(hidden_states.dtype).min
                additive_attention = additive_attention + am

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = (
                True
                if self.training and (dropout_probability < self.config.layerdrop)
                else False
            )
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        additive_attention,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=additive_attention,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
