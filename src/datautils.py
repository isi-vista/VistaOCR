import torch
from torch.utils.data.sampler import Sampler

class GroupedSampler(Sampler):
    """Dataset is divided into sub-groups, G_1, G_2, ..., G_k
       Samples Randomly in G_1, then moves on to sample randomly into G_2, etc all the way to G_k

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, rand=True, max_items=-1, fixed_rand=False):
        self.size_group_keys = data_source.size_group_keys
        self.size_groups = data_source.size_groups
        self.num_samples = len(data_source)
        self.rand = rand
        self.fixed_rand = fixed_rand
        self.max_items = max_items
        self.rand_perm = dict()

    def __iter__(self):
        n_items = 0
        for g in self.size_group_keys:
            if len(self.size_groups[g]) == 0:
                continue

            if self.fixed_rand:
                if g not in self.rand_perm:
                    self.rand_perm[g] = torch.randperm(len(self.size_groups[g])).long()
                g_idx_iter = iter(self.rand_perm[g])
            else:
                if self.rand:
                    g_idx_iter = iter(torch.randperm(len(self.size_groups[g])).long())
                else:
                    g_idx_iter = iter(range(len(self.size_groups[g])))

            while True:
                try:
                    g_idx = next(g_idx_iter)
                except StopIteration:
                    break

                n_items += 1
                if self.max_items > 0 and n_items > self.max_items:
                    raise StopIteration
                yield self.size_groups[g][g_idx]

        raise StopIteration

    def __len__(self):
        return self.num_samples


def SortByWidthCollater(batch):
    return SortByWidthCollater_helper(batch, seq2seq=False, bylang=False)
def SortByWidthCollater_seq2seq(batch):
    return SortByWidthCollater_helper(batch, seq2seq=True, bylang=False)
def SortByWidthCollater_bylang(batch):
    return SortByWidthCollater_helper(batch, seq2seq=False, bylang=True)

def SortByWidthCollater_helper(batch, seq2seq, bylang):
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
    input_langs = []
    input_langs_set = set()

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

        if 'lang' in metadata:
            input_langs.append(metadata['lang'])
            input_langs_set.add(metadata['lang'])

    # Now handle Target Transcripts
    if seq2seq:
        target_transcription = torch.LongTensor(len(batch), target_transcription_widths.max())
        # Init to 0-padding
        target_transcription.fill_(0)
        for idx, (tensor, transcript, metadata) in enumerate(batch):
            for j, char in enumerate(transcript):
                target_transcription[idx][j] = char
    else:
        if bylang:
            # First get counts by lang
            nsamples_by_lang = dict()
            nwords_by_lang = dict()
            for lang in input_langs_set:
                nsamples_by_lang[lang] = 0
                nwords_by_lang[lang] = 0
            for idx, (tensor, transcript, metadata) in enumerate(batch):
                nsamples_by_lang[metadata['lang']] += 1
                nwords_by_lang[metadata['lang']] += target_transcription_widths[idx]
                continue

            # Now let's assign target transcripts for each lang
            target_transcription_by_lang = dict()
            target_transcription_widths_by_lang = dict()
            cur_offset_by_lang = dict()
            idx_by_lang = dict()
            for lang in input_langs_set:
                target_transcription_by_lang[lang] = torch.IntTensor(int(nwords_by_lang[lang]))
                cur_offset_by_lang[lang] = 0
                idx_by_lang[lang] = 0
                target_transcription_widths_by_lang[lang] = torch.IntTensor(nsamples_by_lang[lang])

            for idx, (tensor, transcript, metadata) in enumerate(batch):
                lang = metadata['lang']
                target_transcription_widths_by_lang[lang][idx_by_lang[lang]] = target_transcription_widths[idx]
                idx_by_lang[lang] += 1
                for j, char in enumerate(transcript):
                    target_transcription_by_lang[lang][cur_offset_by_lang[lang]] = char
                    cur_offset_by_lang[lang] += 1

            # Now assign this for convinience
            target_transcription = target_transcription_by_lang
            target_transcription_widths = target_transcription_widths_by_lang
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
    if bylang:
        metadata['input_langs'] = input_langs

    return input_tensor, target_transcription, input_tensor_widths, target_transcription_widths, metadata
