#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: graph_1.2.py
@time: 2021/7/2 8:58
"""
import dgl
import torch as th

g = dgl.graph(([0, 0, 1, 5], [1, 1, 2, 0]))
print(g)
g.ndata["x"] = th.ones(g.num_nodes(), 3)
g.edata["x"] = th.ones(g.num_edges(), dtype=th.int32)
print(g)
# 不同名称的特征可以具有不同形状
g.ndata['y'] = th.randn(g.num_nodes(), 5)
# 获取节点1的特征
print(g.ndata['y'][1])

# 获取边0和3的特征
print(g.edata['x'][th.tensor([0, 3])])

# 成都二环的路段矩阵
u = th.tensor([0, 1, 2, 2, 4, 5, 6, 7, 8, 8, 10, 11, 12, 13])
v = th.tensor([2, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 13, 13, 14])

g = dgl.graph((u, v))

print(g)


if __name__ == "__main__":
    pass
