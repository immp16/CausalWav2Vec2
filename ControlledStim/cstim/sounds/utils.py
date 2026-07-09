import torch
import numpy as np

def shuffle_block(x,block_size : int):
    # Given a nd-tensor x with last dimension the time, this function perform a shuffling
    # along this last dimension.

    # x: nd-tensor
    q = x.shape[-1]//block_size
    r = x.shape[-1]%block_size
    if r>0:
        x = torch.cat([x,torch.zeros(x.shape[:-1]+(block_size-r,),device=x.device,dtype=int)+-1],dim=-1)
        block_frames = torch.reshape(x,x.shape[:-1]+(q+1,block_size))
        orders  = np.random.choice(range(q+1),q+1,replace=False)
    else:
        orders = np.random.choice(range(q),q,replace=False)
        block_frames = torch.reshape(x,x.shape[:-1]+(q,block_size))
    block_frames_out = block_frames[:,:,orders,:]
    block_frames_out = block_frames_out.reshape(block_frames.shape[:-2]+(-1,))
    s = tuple([0 for _ in range(len(block_frames_out.shape[:-1]))]) + (slice(0, block_frames_out.shape[-1]),)
    return block_frames_out[..., torch.logical_not(block_frames_out[s]==-1)]

from packaging import version
parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_less_than_1_8 = parsed_torch_version_base < version.parse("1.8.0")
def get_input_lengths(input_lengths,conv_kernel,conv_stride):

    # This function computes the temporal length of Wav2vec2's latent vector after downsampling by
    # the feature extractor, i.e the set of convolutional layers.
    # It takes as input conv_kernel and conv_stride, a list of the respective kernel siwe and stride of each
    # convolutional layers. These can be obtained from the .conf file along with the huggingface model.


    def torch_int_div(tensor1,tensor2):
        if is_torch_less_than_1_8:
            return tensor1 // tensor2
        else:
            return torch.div(tensor1, tensor2, rounding_mode="floor")
    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return torch_int_div(input_length - kernel_size, stride) + 1

    for kernel_size, stride in zip(conv_kernel, conv_stride):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
    return  input_lengths