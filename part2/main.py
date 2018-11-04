import torch
import torch.nn as nn
import data

def batchify(data, batch_size):
    nBatch = data.size(0) // batch_size
    data = data.narrow(0, 0 , nBatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data

datas = data.load_data()
batch_size = 20
train_data = batchify(datas.train_data, batch_size)
valid_data = batchify(datas.train_data, batch_size)
test_data = batchify(datas.train_data, batch_size)
print(train_data)
