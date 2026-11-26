"""保持 BF16 乘法中间舍入，以线性输出分块执行索引门控。"""
import ninetoothed.language as ntl


def application(residual, activation, gate, indices, output):
    offset = output.offsets(0).to(ntl.int32)
    row = offset // 5376
    column = offset % 5376
    modulation_row = indices.source[row].to(ntl.int32)
    gate_value = gate.source[modulation_row, column].to(ntl.float32)
    branch = activation.source[row, column].to(ntl.float32)
    scaled = (gate_value * branch).to(output.dtype).to(ntl.float32)
    output = residual.source[row, column].to(ntl.float32) + scaled
