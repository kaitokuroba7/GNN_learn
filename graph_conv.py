#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: graph_conv.py
@time: 2021/7/2 10:51
"""
import dgl
import numpy as np
import torch as th
from dgl.nn import GraphConv

g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
g = dgl.add_self_loop(g)
feat = th.ones(6, 10)
conv = GraphConv(10, 2, norm='both', weight=True, bias=True)
res = conv(g, feat)
print(res)

# allow_zero_in_degree example
g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
conv = GraphConv(10, 2, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
res = conv(g, feat)
print(res)

if __name__ == "__main__":
    pass
