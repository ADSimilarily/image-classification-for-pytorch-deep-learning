from torch.utils.data import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, num_iterations, start_iter=0):
        super(IterationBasedBatchSampler, self).__init__(sampler, batch_size, False)
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    iteration += 1
            if len(batch) >= 8 and not self.drop_last:
                yield batch
                iteration += 1

    def __len__(self):
        return self.num_iterations