"""动态长度 packed BF16 SwiGLU，保留原库 sigmoid BF16 舍入及快速倒数。"""
import ninetoothed.language as ntl


def application(up,gate,output):
    index=output.offsets(0).to(ntl.int32)
    row=index//1792
    column=index%1792
    u=up.source[row,column].to(ntl.float32)
    g=gate.source[row,column].to(ntl.float32)
    sig=(ntl.libdevice.fast_dividef(1.0,1.0+ntl.libdevice.fast_expf(-g))).to(output.dtype).to(ntl.float32)
    output=(g*sig)*u
