

class BatchCollector(object):
    def __init__(self):
        self.name = "BatchCollector"

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = transposed_batch[0]
        targets = transposed_batch[1]
        return images, targets