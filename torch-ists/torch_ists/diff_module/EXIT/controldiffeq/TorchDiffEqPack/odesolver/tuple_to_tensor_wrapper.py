"""
This file provides an example to convert a function with multiple inputs to a function with a single tensor output.
Because the solver currently only supports tensor input, for the ease of memory saving
"""
import torch.nn as nn
import torch
import numpy as np

def tuple_to_tensor( inputs):
    shapes = [ [1] + list(_input.shape)[1:] for _input in inputs]
    concats = [_input.view(_input.shape[0], -1) for _input in inputs]  # reshape to shape batch x -1
    concats = torch.cat(concats, -1)
    return shapes, concats # N x -1

def tensor_to_tuple( shapes, concats):
    outs = []
    tmp = 0
    for shape in shapes:
        _size = int(np.prod(list(shape)[1:]))
        out = concats[:, tmp:tmp + _size]
        tmp += _size
        outs.append(out.view(shape))
    return tuple(outs)


class TupleFuncToTensorFunc(nn.Module):
    def __init__(self, func, shapes):
        """
        :param func: func takes tuple inputs, tuple outputs
        :param shapes:
        """
        super(TupleFuncToTensorFunc, self).__init__()
        self.func = func
        self.shapes = shapes

    def forward(self, t, x):
        input_tensors = tensor_to_tuple(self.shapes, x)
        outs = self.func(t, input_tensors)
        shapes, concats = tuple_to_tensor(outs)
        return concats



