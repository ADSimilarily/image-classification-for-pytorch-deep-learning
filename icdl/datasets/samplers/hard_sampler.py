
class HardSampler:

    sampler_dict = {}
    epoch_count = 0

    @classmethod
    def clearSamplerDict(cls):
        cls.sampler_dict.clear()

    @classmethod
    def epoch(cls, epoch):
        cls.epoch_count = epoch

    @classmethod
    def getRandomSampler(cls):
        if not cls.sampler_dict or cls.epoch_count % 2 == 0:
            return None
        else:
            return cls.sampler_dict

    @classmethod
    def insertSampler(cls, key, value):
        cls.sampler_dict[key] = value