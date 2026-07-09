import transformers
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining, set_seed, get_scheduler, is_wandb_available, SchedulerType
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
import torch
from torch.utils.data.dataloader import DataLoader
import math
from dataclasses import dataclass
from accelerate import Accelerator
from accelerate.logging import get_logger
import datasets
from datasets import load_dataset, Audio, concatenate_datasets, DatasetDict
from surprise_probing.ANN.models.wav2vec2.with_grad_mult import Wav2Vec2ForPreTraining_latentMask_izei
from tqdm.auto import tqdm
from typing import Optional, Union
import os
import argparse
import wandb

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain wav2vec on any dataset")
    parser.add_argument(
        "--dataset_names",
        nargs='+',
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_names",
        nargs="+",
        type=str,
        required=True,
        help="The configuration names of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split_names",
        nargs="+",
        type=str,
        required=True,
        help="The names of the training data set splits to use (via the datasets library).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--dataset_cache_dirs",
        nargs="+",
        type=str,
        required=True,
        help="The paths to the downloaded files",
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=1,
        help="Percentage of training data that should be used for validation if no validation is present in dataset.",
    )
    parser.add_argument(
        "--logging_completed_updates",
        type=int,
        default=1,
        help="Number of update steps between each logging",
    )
    parser.add_argument(
        "--saving_completed_updates",
        type=int,
        default=100,
        help="Number of update steps between each saving",
    )
    parser.add_argument(
        "--ending_updates",
        type=int,
        default=5000,
        help="Number of update steps until the process dies",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="patrickvonplaten/wav2vec2-base-v2",
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="patrickvonplaten/wav2vec2-base-v2",
        help="Pretrained config name",
    )
    parser.add_argument(
        "--diversity_loss_weight",
        type=float,
        default=0.25,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--features_pen_weight",
        type=float,
        default=0.01,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_updates",
        type=int,
        default=200000,
        help="Total number of training updates to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_updates", type=int, default=32000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--checkpoint_name", type=str, default='wav2vec2.checkpoint', help="Where to load the model.")
    parser.add_argument("--output_checkpoint_name", type=str, default='wav2vec2.checkpoint', help="Where to store the model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_gumbel_temperature",
        type=float,
        default=2.0,
        help="Maximum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--min_gumbel_temperature",
        type=float,
        default=0.5,
        help="Minimum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--gumbel_temperature_decay", type=float, default=0.999995, help="Decay of gumbel temperature during training."
    )
    parser.add_argument(
        "--gumbel_annealing", type=int, default=1, help="Number of update steps where temperature will not be updated."
    )
    parser.add_argument(
        "--max_duration_in_seconds",
        type=float,
        default=20.0,
        help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--min_duration_in_seconds",
        type=float,
        default=2.0,
        help="Filter out audio files that are shorter than `min_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help=(
            "If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the"
            " use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        ),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.98,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-06,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument(
        "--mask_time_prob",
        type=float,
        default=0.65,
        help=(
            "Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked in the"
            " contrastive task. If omitted, will pull value from model config."
        ),
    )
    parser.add_argument(
        "--mask_time_length",
        type=int,
        default=10,
        help=(
            "Length of each vector mask span to mask along the time axis in the contrastive task."
            " If omitted, will pull value from model config."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        default='bf16'
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default='c81ed98e1978e50aacf941de92e34b69c7bee0b7'
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default='wav2vec2'
    )
    parser.add_argument(
        "--wandb_project_id",
        type=str,
        default='default'
    )
    parser.add_argument(
        "--resume_training",
        type=str,
        default='False'
    )

    args = parser.parse_args()
    return args

@dataclass
class DataCollatorForWav2Vec2Pretraining:
  model: Union[Wav2Vec2ForPreTraining_latentMask_izei, Wav2Vec2ForPreTraining]
  precision: torch.dtype
  feature_extractor: Wav2Vec2FeatureExtractor
  padding: Union[bool, str] = "longest"
  pad_to_multiple_of: Optional[int] = None
  mask_time_prob: Optional[float] = 0.65
  mask_time_length: Optional[int] = 10
  max_duration: float = 20.0

  def __call__(self, examples: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:

      arrays = [ex["audio"]["array"] for ex in examples]
      features = self.feature_extractor(
          arrays,
          sampling_rate=self.feature_extractor.sampling_rate,
          max_length=int(self.max_duration * self.feature_extractor.sampling_rate),
          truncation=True,
          return_attention_mask=True
      )

      # Apply the preprocessor
      batch = self.feature_extractor.pad(
        features, # Numpy arrays
        padding=self.padding,
        pad_to_multiple_of=self.pad_to_multiple_of,
        return_tensors="pt", # Pytorch-like tensors, as in a tokenizer
      ).to(self.precision)

      # Obtain batch size and device of the batch.
      device = batch["input_values"].device
      batch_size = batch["input_values"].shape[0]

      # Obtain the length of the output of the convolutional layer, size of the encoded vectors.
      mask_indices_seq_length = self.model.wav2vec2._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
      mask_indices_seq_length = int(mask_indices_seq_length) # Make sure masked sequence length is a Python scalar

      # Make sure that no loss is computed on padded inputs. When padding occurs, there
      # are some tokens in the end that we don't need to consider in any form when training.
      if batch.get("attention_mask") is not None:
        # Compute real output lengths according to convolution formula
        sub_attention_mask = self.model.wav2vec2._get_feature_vector_attention_mask(
          mask_indices_seq_length, batch["attention_mask"]
        ) # Returns the attention mask we would apply to the vector of features.
          # It's like resizing the attention mask so that it covers the latent
          # representations of the padding tokens.
        batch["sub_attention_mask"] = sub_attention_mask

      features_shape = (batch_size, mask_indices_seq_length)

      # Sample randomly masked indices
      mask_time_indices = _compute_mask_indices(
        features_shape,
        self.mask_time_prob,
        self.mask_time_length,
        attention_mask=batch.get("sub_attention_mask"),
      )

      # Sample negative indices
      sampled_negative_indices = _sample_negative_indices(
        features_shape,
        self.model.config.num_negatives,
        mask_time_indices=mask_time_indices,
      )

      # Generate the batch to be used during training.
      batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
      batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

      return batch

def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm

def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)

def cast16k(ds, feature_extractor):
    return ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))

def main():
    args = parse_args()
    home_dir = os.path.expanduser("~/")

    ############################ Initialize accelerator and seed ############################
    accelerator = Accelerator()
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()

        if is_wandb_available():
            wandb.login(key = args.wandb_api_key)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    accelerator.wait_for_everyone()

    if args.seed is not None:
      set_seed(args.seed)
    #########################################################################################

    ########################## Configuration and feature extractor ##########################
    config = Wav2Vec2Config.from_pretrained(args.model_name)
    config.diversity_loss_weight = args.diversity_loss_weight
    config.features_pen_weight = args.features_pen_weight
    config.mask_time_prob = config.mask_time_prob if args.mask_time_prob is None else args.mask_time_prob
    config.mask_time_length = config.mask_time_length if args.mask_time_length is None else args.mask_time_length

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.config_name)
    #########################################################################################

    ##################################### Load datasets #####################################
    datasets_splits = []
    for dataset_name, dataset_config_name, train_split_name, cache_dir in zip(args.dataset_names, args.dataset_config_names, args.dataset_split_names, args.dataset_cache_dirs):
        # load dataset
        dataset_split = load_dataset(
            dataset_name,
            dataset_config_name,
            split=train_split_name,
            cache_dir=f'{home_dir}datasets/{cache_dir}',
            trust_remote_code=args.trust_remote_code,
        )
        dataset_split = cast16k(dataset_split, feature_extractor)
        datasets_splits.append(dataset_split)

    all_datasets = DatasetDict()
    all_datasets['train'] = concatenate_datasets(datasets_splits).shuffle(seed=args.seed)
    num_test_samples = all_datasets["train"].num_rows * args.validation_split_percentage // 100
    all_datasets["test"] = all_datasets["train"].select(range(num_test_samples))
    all_datasets["train"] = all_datasets["train"].select(range(num_test_samples, all_datasets["train"].num_rows))
    #########################################################################################

    ##################### Load model, collator, optimizer and scheduler #####################
    model = Wav2Vec2ForPreTraining_latentMask_izei(config)
    mask_time_prob = config.mask_time_prob
    mask_time_length = config.mask_time_length

    precision = torch.float32
    data_collator = DataCollatorForWav2Vec2Pretraining(
      model=model,
      precision=precision,
      feature_extractor=feature_extractor,
      pad_to_multiple_of=None,
      mask_time_prob=mask_time_prob,
      mask_time_length=mask_time_length,
      max_duration = args.max_duration_in_seconds
    )

    train_dataloader = DataLoader(
        all_datasets['train'],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )

    eval_dataloader = DataLoader(
        all_datasets['test'],
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.lr,
        betas=[args.beta1, args.beta2],
        eps=args.eps,
        weight_decay = args.weight_decay
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps * accelerator.num_processes)) # Because we are doing before accelerate.prepare()!!! So, we need to consider that each GPU will process len(train_dataloader) / num_GPUs!

    if args.max_train_updates is None:
        num_train_epochs = 3
        args.max_train_updates = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_updates,
        num_training_steps=args.max_train_updates,
    )

    resume = False if args.resume_training == 'False' else True
    if resume:
        checkpoint = torch.load(f'{home_dir}pretraining/{args.checkpoint_name}', weights_only=False)
        model.load_state_dict(checkpoint['model'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        current_step = checkpoint['current_step']
        current_epoch = checkpoint['epoch']
        args.max_train_updates = checkpoint['max_train_updates']
        completed_updates = checkpoint['completed_updates']
        lr_scheduler.step(completed_updates)

        gumbel_temperature = checkpoint['gumbel_temperature']
        model.set_gumbel_temperature(gumbel_temperature)

        if accelerator.is_local_main_process:
            if is_wandb_available():
                wandb.init(project = args.wandb_project_name, id = args.wandb_project_id, name = args.wandb_project_id, resume='must')
    else:
        if accelerator.is_local_main_process:
            if is_wandb_available():
                wandb.init(project = args.wandb_project_name, id = args.wandb_project_id, name = args.wandb_project_id)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = math.ceil(args.max_train_updates / num_update_steps_per_epoch)
    #########################################################################################

    ################################ Training and validating ################################
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    end = False

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(all_datasets['train'])}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_updates}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_updates), disable=not accelerator.is_local_main_process)
    if not resume:
      completed_updates = 0
      starting_epoch = current_epoch = 0
      current_step = -1
    else:
      starting_epoch = current_epoch
      progress_bar.update(completed_updates)

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if epoch * len(train_dataloader) + step <= current_epoch * len(train_dataloader) + current_step:
                continue

            num_losses = batch["mask_time_indices"].sum()
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()

            # Forward
            outputs = model(**batch)

            is_last_batch = (step == len(train_dataloader) - 1)
            leftover = (step % args.gradient_accumulation_steps) + 1
            loss_scale = args.gradient_accumulation_steps if (not is_last_batch or leftover == args.gradient_accumulation_steps) else leftover

            # Divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = outputs.loss / loss_scale
            accelerator.backward(loss)

            if accelerator.state.num_processes > 1:
              num_losses = accelerator.gather_for_metrics(num_losses).sum()
              gradient_multiplier = accelerator.state.num_processes / num_losses
              multiply_grads(model.module.parameters(), gradient_multiplier)
            else:
              multiply_grads(model.parameters(), 1 / num_losses)

            # Update step
            if (step + 1) % args.gradient_accumulation_steps == 0 or is_last_batch:
                if accelerator.state.num_processes > 1:
                  torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
                else:
                  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scale = (
                  accelerator.scaler._scale.item()
                  if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                  else 1
                )
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.parameters(), scale)

                # Update parameters
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )

                # Update gumbel temperature
                gumbel_temperature = args.max_gumbel_temperature if completed_updates <= args.gumbel_annealing else max(
                    args.max_gumbel_temperature * args.gumbel_temperature_decay**completed_updates,
                    args.min_gumbel_temperature,
                )
                if hasattr(model, "module"):
                    model.module.set_gumbel_temperature(gumbel_temperature)
                else:
                    model.set_gumbel_temperature(gumbel_temperature)

                progress_bar.update(1)
                completed_updates += 1

            # Log all results
            if completed_updates and (completed_updates % args.logging_completed_updates == 0) and ((step + 1) % args.gradient_accumulation_steps == 0 or is_last_batch):
                loss.detach()
                outputs.contrastive_loss.detach()
                outputs.diversity_loss.detach()

                if accelerator.state.num_processes > 1:
                  loss = accelerator.gather_for_metrics(loss).sum()
                  outputs.contrastive_loss = accelerator.gather_for_metrics(outputs.contrastive_loss).sum()
                  outputs.diversity_loss = accelerator.gather_for_metrics(outputs.diversity_loss).sum()
                  outputs.pen_loss = accelerator.gather_for_metrics(outputs.pen_loss).sum()
                  percent_masked = accelerator.gather_for_metrics(percent_masked).sum()

                train_logs = {
                    "Loss": (loss * loss_scale) / num_losses,
                    "Contrastive loss": outputs.contrastive_loss / num_losses,
                    "Diversity loss": outputs.diversity_loss / num_losses,
                    "Feature penalization loss": outputs.pen_loss / num_losses,
                    "Mask indices (%)": percent_masked / accelerator.num_processes,
                    "Perplexity": outputs.codevector_perplexity,
                    "Learning rate": torch.tensor(optimizer.param_groups[0]["lr"]),
                    "Gumbel temperature": torch.tensor(gumbel_temperature),
                    "Gradient norm": torch.tensor(grad_norm),
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += f"| {k}: {v.item():.3e}"

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)
                    if is_wandb_available():
                        wandb.log(train_logs)

            # Save model every 'saving_steps' steps (considering full steps, parameter updates)
            if completed_updates and (completed_updates % args.saving_completed_updates == 0) and ((step + 1) % args.gradient_accumulation_steps == 0 or is_last_batch):
              accelerator.wait_for_everyone()
              unwrapped_model = accelerator.unwrap_model(model)

              checkpoint = {
                'epoch': epoch,
                'model': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'current_step': step,
                'gumbel_temperature': gumbel_temperature,
                'max_train_updates': args.max_train_updates,
                'completed_updates': completed_updates
              }
              torch.save(checkpoint, f'{home_dir}pretraining/{args.output_checkpoint_name}')
              if completed_updates % args.ending_updates == 0:
                  end = True
                  break

            # if completed updates > 'args.max_train_updates' stop
            if completed_updates >= args.max_train_updates:
                end = True
                break

        if end:
            break

        # Validation
        model.eval()

        # init logs
        val_logs = {
            "Validation loss": 0,
            "Validation contrastive loss": 0,
            "Validation diversity loss": 0,
            "Validation feature penalization loss": 0,
            "Validation number losses": 0,
        }
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)

            val_logs["Validation loss"] += outputs.loss
            val_logs["Validation contrastive loss"] += outputs.contrastive_loss
            val_logs["Validation diversity loss"] += outputs.diversity_loss
            val_logs["Validation feature penalization loss"] += outputs.pen_loss
            val_logs["Validation number losses"] += batch["mask_time_indices"].sum()

        val_logs = {k: v / val_logs["Validation number losses"] for k, v in val_logs.items()}

        log_str = ""
        for k, v in val_logs.items():
            log_str += f"| {k}: {v.item():.3e}"

        if accelerator.is_local_main_process:
            progress_bar.write(log_str)
            if is_wandb_available():
                wandb.log(val_logs)

    if completed_updates >= args.max_train_updates:
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint = {
            'epoch': num_train_epochs,
            'model': unwrapped_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'current_step': 0,
            'gumbel_temperature': gumbel_temperature,
            'max_train_updates': args.max_train_updates,
            'completed_updates': completed_updates
        }
        torch.save(checkpoint, f'{home_dir}pretraining/{args.output_checkpoint_name}')

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
