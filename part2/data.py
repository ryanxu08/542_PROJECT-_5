import torch
from torchnlp.datasets import penn_treebank_dataset as ptb
from torchnlp.text_encoders import IdentityEncoder
from torchnlp.samplers import BPTTBatchSampler

# load_data


class load_data(object):
    def __init__(self,is_char = False):
        if not is_char:
            self.train, self.valid, self.test = ptb(train=True, dev = True, test=True)
        else:
            self.train, self.valid, self.test = ptb(train=True, dev = True, test=True, train_filename="ptb.char.train.txt",
                                                    dev_filename = "ptb.char.valid.txt",test_filename="ptb.char.test.txt")
        self.train_data = self._encode_data(self.train)
        self.valid_data = self._encode_data(self.valid)
        self.test_data  = self._encode_data(self.test)
    def _encode_data(self, input):
        data = dict()
        for word in input:
            if word not in data.keys():
                data[word] = len(data)
        idx = torch.LongTensor(len(input))
        for index, word  in enumerate(input):
            idx[index] = data[word]
        return idx

