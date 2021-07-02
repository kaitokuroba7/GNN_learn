#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: graph.py
@time: 2021/7/2 8:27
"""
import dgl
import torch as th

u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))
print(g)
# 获得边的对应端
print(g.edges())
# 获取边的对应端和边的ID
print(g.edges(form="all"))

# g = dgl.graph((u, v), num_nodes=8)
# print(g)

# 无向图
bg = dgl.to_bidirected(g)
print(bg.edges())

if __name__ == "__main__":
    pass
