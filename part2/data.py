import torch
from torchnlp.datasets import penn_treebank_dataset as ptb
from torchnlp.text_encoders import IdentityEncoder

# load_data


class Load_data(object):
    def __init__(self,is_char = False):
        self.mapped_data = dict()
        if not is_char:
            self.train, self.valid, self.test = ptb(train=True, dev = True, test=True)
        else:
            self.train, self.valid, self.test = ptb(train=True, dev = True, test=True, train_filename="ptb.char.train.txt",
                                                    dev_filename = "ptb.char.valid.txt",test_filename="ptb.char.test.txt")
        self._map_data(self.train + self.valid + self.test)
        #encodeer data
        encoder = IdentityEncoder(self.train + self.valid + self.test)
        self.train = torch.LongTensor(encoder.encode(self.train))
        self.valid = torch.LongTensor(encoder.encode(self.valid))
        self.test = torch.LongTensor(encoder.encode(self.test))
        self.ntoken = encoder.vocab_size
        print('hello')

    def _map_data(self, input):
        for word in input:
            if word not in self.mapped_data.keys():
                self.mapped_data[word] = len(self.mapped_data.keys()) +5

