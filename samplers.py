import torch
import torch.utils.data
from torch._six import int_classes as _int_classes
from random import shuffle
from collections import OrderedDict


class OnlineHardBatchSampler(torch.utils.data.Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, data_source, batch_size, drop_last):
        if not isinstance(data_source, torch.utils.data.Dataset):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(data_source))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.ranks = OrderedDict()
        self.rank_idx = 0

        for idx in range(len(self.data_source)):
            self.ranks[idx] = float('inf')

        self.ranks = list(self.ranks.items())
        shuffle(self.ranks)
        self.ranks = OrderedDict(self.ranks)

    def __iter__(self):
        batch = []
        self.rank_idx = 0
        for idx in range(len(self.data_source)):
            batch.append(list(self.ranks.keys())[self.rank_idx])
            self.rank_idx += 1
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def set_batch_losses(self, batch_indicies_and_losses):
        for idx, loss in batch_indicies_and_losses:
            self.ranks[idx] = loss

        self.sort()

    def sort(self):
        self.ranks = OrderedDict(sorted(self.ranks.items(), key=lambda item: item[1], reverse=True))
        # self.ranks.sort(key=lambda x: x[0], reverse=True)
        self.rank_idx = 0

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size
