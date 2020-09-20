from typing import List

import numpy as np
from torch.utils.data.sampler import Sampler
from icdl.datasets.samplers.hard_sampler import HardSampler
import itertools
from icdl.utils import comm


class BalancedIdentitySampler(Sampler):
    """
    BatchSampler - from a dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, labels: List[int], n_classes: int, n_samples: int):
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

        for label in self.labels_set:
            np.random.shuffle(self.label_to_indices[label])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples * comm.get_world_size()
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

        seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def update_label_indices(self, indices):
        self.label_to_indices = indices

    def _get_epoch_indices(self):
        self.count = 0

        while self.count + self.batch_size < self.n_dataset:
            indices = []
            classes = HardSampler.getRandomSampler()      # 困难类别挖掘
            if classes is None:
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            else:
                x = np.random.choice(list(classes.keys()))
                classes = classes[x]
            for class_ in classes:
                sub_indice = []

                # 确保每类的数量是self.n_samples
                while len(sub_indice) < self.n_samples:
                    need_count = self.n_samples - len(sub_indice)    # 需要加入的量
                    used_count = self.used_label_indices_count[class_]
                    add_indice = self.label_to_indices[class_][used_count:used_count+need_count]
                    sub_indice += list(add_indice)

                    self.used_label_indices_count[class_] += need_count

                    if self.used_label_indices_count[class_] + need_count > \
                            len(self.label_to_indices[class_]):
                        np.random.shuffle(self.label_to_indices[class_])
                        self.used_label_indices_count[class_] = 0

                indices.extend(sub_indice)
                assert len(sub_indice) == self.n_samples
            yield from indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        indices = self._get_epoch_indices()
        return indices
