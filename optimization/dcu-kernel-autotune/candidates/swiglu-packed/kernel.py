"""MiniMax-H3 TP=8 的动态长度 BF16 SwiGLU；保留 sigmoid 的 BF16 舍入。"""
import ninetoothed.language as ntl


def arrange(up,gate,output):
    return up.tile((1,1792)),gate.tile((1,1792)),output.tile((1024,))


def application(up,gate,output):
    index=output.offsets(0).to(ntl.int32)
    row=index//1792
    column=index%1792
    u=up.source[row,column].to(ntl.float32)
    g=gate.source[row,column].to(ntl.float32)
    sig=(1.0/(1.0+ntl.exp(-g))).to(output.dtype).to(ntl.float32)
    output=(g*sig)*u


def application_fast(up,gate,output):
    index=output.offsets(0).to(ntl.int32)
    row=index//1792
    column=index%1792
    u=up.source[row,column].to(ntl.float32)
    g=gate.source[row,column].to(ntl.float32)
    sig=(1.0/(1.0+ntl.libdevice.fast_expf(-g))).to(output.dtype).to(ntl.float32)
    output=(g*sig)*u


def application_fast_rcp(up,gate,output):
    index=output.offsets(0).to(ntl.int32)
    row=index//1792
    column=index%1792
    u=up.source[row,column].to(ntl.float32)
    g=gate.source[row,column].to(ntl.float32)
    sig=(ntl.libdevice.fast_dividef(1.0,1.0+ntl.libdevice.fast_expf(-g))).to(output.dtype).to(ntl.float32)
    output=(g*sig)*u


def application_native_exp(up,gate,output):
    index=output.offsets(0).to(ntl.int32)
    row=index//1792
    column=index%1792
    u=up.source[row,column].to(ntl.float32)
    g=gate.source[row,column].to(ntl.float32)
    x=-g
    high=x*1.4426950216293335
    low=ntl.libdevice.fma(x,1.4426950216293335,-high)
    low=ntl.libdevice.fma(x,1.925962855864327e-08,low)
    power=ntl.libdevice.nearbyint(high)
    fraction=(high-power)+low
    exponent=ntl.libdevice.ldexp(ntl.exp2(fraction),power.to(ntl.int32))
    exponent=ntl.where(x<-103.2789306640625,0.0,exponent)
    exponent=ntl.where(x>88.72283935546875,1e39,exponent).to(ntl.float32)
    sig=ntl.libdevice.fast_dividef(1.0,1.0+exponent).to(output.dtype).to(ntl.float32)
    output=(g*sig)*u
