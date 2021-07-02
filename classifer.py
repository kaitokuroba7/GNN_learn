#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: classifer.py
@time: 2021/7/2 19:57
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data

dataset = dgl.data.CoraGraphDataset()
print('Number of categories:', dataset.num_classes)

g = dataset[0]
print(g)

print('Node features')
print(g.ndata)
print('Edge features')
print(g.edata)

if __name__ == "__main__":
    pass
