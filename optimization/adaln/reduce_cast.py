"""通信精度实验的九齿浮点转换；权重与 GEMM 计算保持原实现。"""
import ninetoothed.language as ntl


def arrange(input, output):
    return input.tile((1024,)), output.tile((1024,))


def application(input, output):
    output = input


def scale_down_for_wire(input, output):
    output = input.to(ntl.float32) * (1.0 / 256.0)


def restore_from_wire(input, output):
    output = input.to(ntl.float32) * 256.0
