import torch
import torch.nn as nn


class ONNX:
    onnx = False

    @classmethod
    def onnx_export(cls, on):
        cls.onnx = on

    @classmethod
    def use_onnx(cls):
        return cls.onnx


class OnnxBatchNorm1d(nn.Module):
    def __init__(self, bn):
        super(OnnxBatchNorm1d, self).__init__()
        self.eps = bn.eps
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var
        self.weight = bn.weight
        self.bias = bn.bias

    def forward(self, x):
        y = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        out = self.weight * y + self.bias
        return out


class OnnxAvgPool2d(nn.Module):
    def __init__(self, in_channel, kernel, stride, padding=0):
        super(OnnxAvgPool2d, self).__init__()
        self.in_channel = in_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = torch.ones(in_channel, 1, kernel, kernel)

    def forward(self, inputs):
        x = nn.functional.conv2d(inputs, self.weight, None, self.stride, self.padding, groups=self.in_channel)
        return x / (self.kernel * self.kernel)


class OnnxAvgCeilPool2d(nn.Module):
    def __init__(self, in_channel, kernel, stride, padding=0):
        super(OnnxAvgCeilPool2d, self).__init__()
        self.in_channel = in_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = torch.ones(in_channel, 1, kernel, kernel)
        self.const_pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, inputs):
        inputs = self.const_pad(inputs)
        x = nn.functional.conv2d(inputs, self.weight, None, self.stride, self.padding, groups=self.in_channel)
        x0 = x[:, :, -1, -1]
        x1 = x[:, :, :-1, :-1] / (self.kernel * self.kernel)
        x = x / (self.kernel * self.kernel) * 2
        x[:, :, :-1, :-1] = x1
        x[:, :, -1, -1] = x0
        return x