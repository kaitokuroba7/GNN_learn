#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: graph_gpu.py
@time: 2021/7/2 10:32
"""
import dgl
import torch as th

u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
g = dgl.graph((u, v))
g.ndata["x"] = th.randn(5, 3)
print(g.device)

# to cuda
cuda_g = g.to("cuda:0")
print(cuda_g.device)

print(cuda_g.ndata['x'].device)

if __name__ == "__main__":
    pass
