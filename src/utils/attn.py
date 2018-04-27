import torch


# Used util code from open-nmt code:  https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Utils.py
def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if type(lengths) == torch.autograd.Variable:
        lengths_ = lengths.data
    else:
        lengths_ = lengths

    batch_size = lengths_.numel()
    max_len = max_len or lengths_.max()

    return (torch.arange(0, max_len)
            .type_as(lengths_)
            .repeat(batch_size, 1)
            .lt(lengths_.unsqueeze(1)))

