import torch
from torch.utils.data.sampler import Sampler

class GroupedSampler(Sampler):
    """Dataset is divided into sub-groups, G_1, G_2, ..., G_k
       Samples Randomly in G_1, then moves on to sample randomly into G_2, etc all the way to G_k

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, rand=True):
        self.size_group_keys = data_source.size_group_keys
        self.size_groups = data_source.size_groups
        self.num_samples = len(data_source)
        self.rand = rand

    def __iter__(self):
        for g in self.size_group_keys:
            if len(self.size_groups[g]) == 0:
                continue

            if self.rand:
                g_idx_iter = iter(torch.randperm(len(self.size_groups[g])).long())
            else:
                g_idx_iter = iter(range(len(self.size_groups[g])))

            while True:
                try:
                    g_idx = next(g_idx_iter)
                except StopIteration:
                    break

                yield self.size_groups[g][g_idx]
        raise StopIteration

    def __len__(self):
        return self.num_samples


def SortByWidthCollater(batch):
    return SortByWidthCollater_helper(batch, seq2seq=False)
def SortByWidthCollater_seq2seq(batch):
    return SortByWidthCollater_helper(batch, seq2seq=True)

def SortByWidthCollater_helper(batch, seq2seq):
    # Takes in a list of (InputTensor, TargetTranscriptArray) tuples
    # Sort these by width in decreasing order and form into single Tensor
    #
    # Want to return InputTensor, TargetTensor, InputWidths, TargetWidths
    #  Where both InputTensor and TargetTensor have to be padded out to size of largest entry
    #  So the actual width counts (w/o padding) are also provided as additional output

    _use_shared_memory = True

    # Sort by tensor width
    batch.sort(key=lambda d: d[-1]['width'], reverse=True)

    # Deal with use_shared_memory???
    # Does this to extra copies???
    biggest_size = batch[0][0].size()
    input_tensor = torch.zeros(len(batch), biggest_size[0], biggest_size[1], biggest_size[2])
    input_tensor_widths = torch.IntTensor(len(batch))
    target_transcription_widths = torch.IntTensor(len(batch))
    writer_ids = torch.LongTensor(len(batch))

    wid_feats = None
    sample_ids = []
    include_writers = False
    include_ids = False

    for idx, (tensor, transcript, metadata) in enumerate(batch):
        width = tensor.size(2)
        input_tensor[idx, :, :, :width] = tensor
        input_tensor_widths[idx] = metadata['width']  # image may come to us already padded
        target_transcription_widths[idx] = len(transcript)

        if 'writer-id' in metadata:
            writer_ids[idx] = metadata['writer-id']
            include_writers = True

        if 'utt-id' in metadata:
            sample_ids.append(metadata['utt-id'])
            include_ids = True

        if 'wid-feat' in metadata:
            if wid_feats is None:
                wid_feats = torch.zeros(len(batch), metadata['wid-feat'].size(0))

            wid_feats[idx] = metadata['wid-feat']

    # Now handle Target Transcripts
    if seq2seq:
        target_transcription = torch.LongTensor(len(batch), target_transcription_widths.max())
        # Init to 0-padding
        target_transcription.fill_(0)
        for idx, (tensor, transcript, metadata) in enumerate(batch):
            for j, char in enumerate(transcript):
                target_transcription[idx][j] = char
    else:
        target_transcription = torch.IntTensor(target_transcription_widths.sum().item())
        cur_offset = 0
        for idx, (tensor, transcript, metadata) in enumerate(batch):
            for j, char in enumerate(transcript):
                target_transcription[cur_offset] = char
                cur_offset += 1

    metadata = {}
    if include_writers:
        metadata['writer-ids'] = writer_ids
    if include_ids:
        metadata['utt-ids'] = sample_ids
    if not wid_feats is None:
        metadata['wid-feats'] = wid_feats

    return input_tensor, target_transcription, input_tensor_widths, target_transcription_widths, metadata
