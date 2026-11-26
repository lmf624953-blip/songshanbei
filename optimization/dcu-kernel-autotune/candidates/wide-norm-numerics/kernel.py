"""只导出归约和 inverse RMS 中间量，定位九齿融合的舍入差异。"""
import ninetoothed.language as ntl


def arrange(hidden, weight, normalized, sums, inverse):
    h=hidden.tile((1,8192))
    out=normalized.tile((1,8192))
    w=weight.tile((8192,)).unsqueeze(0).expand(out.shape)
    return h,w,out,sums.tile((1,1)),inverse.tile((1,1))


def naive(hidden, weight, normalized, sums, inverse):
    x=hidden.to(ntl.float32)
    ss=ntl.sum(x*x,-1)
    inv=ntl.rsqrt(ss/5376+1e-5)
    sums=ss[:,None]
    inverse=inv[:,None]
    normalized=(x*weight.to(ntl.float32))*inv[:,None]


def cub_warp(hidden, weight, normalized, sums, inverse):
    token=normalized.offsets(0).to(ntl.int32)
    column=normalized.offsets(1).to(ntl.int32)
    d=((column&1)<<3)+((column&2)<<1)+((column&4)>>1)+((column&8)>>3)+(column&1008)
    d=(d&63)+((d&64)<<3)+((d&128)<<1)+((d&256)>>1)+((d&512)>>3)
    x0=hidden.source[token,d].to(ntl.float32)
    x1=hidden.source[token,d+1024].to(ntl.float32)
    x2=hidden.source[token,d+2048].to(ntl.float32)
    x3=hidden.source[token,d+3072].to(ntl.float32)
    x4=hidden.source[token,d+4096].to(ntl.float32)
    x5=hidden.source[token,(d&255)+5120].to(ntl.float32)
    partial=(((x0*x0+x1*x1)+x2*x2)+x3*x3)+x4*x4
    partial=partial+ntl.where(d<256,x5*x5,0.0)
    partial=ntl.where(column<1024,partial,0.0)+ntl.zeros(normalized.shape,dtype=ntl.float32)
    ss=ntl.sum(partial,-1)
    inv=ntl.rsqrt(ss/5376+1e-5)
    sums=ss[:,None]
    inverse=inv[:,None]
    x=hidden.to(ntl.float32)
    normalized=(x*weight.to(ntl.float32))*inv[:,None]


def original_serial(hidden, weight, normalized, sums, inverse):
    token=normalized.offsets(0).to(ntl.int32)
    column=normalized.offsets(1).to(ntl.int32)
    d=((column&1)<<3)+((column&2)<<1)+((column&4)>>1)+((column&8)>>3)+(column&1008)
    x0=hidden.source[token,d].to(ntl.float32)
    x1=hidden.source[token,d+1024].to(ntl.float32)
    x2=hidden.source[token,d+2048].to(ntl.float32)
    x3=hidden.source[token,d+3072].to(ntl.float32)
    x4=hidden.source[token,d+4096].to(ntl.float32)
    x5=hidden.source[token,(d&255)+5120].to(ntl.float32)
    partial=(((x0*x0+x1*x1)+x2*x2)+x3*x3)+x4*x4
    partial=partial+ntl.where(d<256,x5*x5,0.0)
    partial=ntl.where(column<1024,partial,0.0)+ntl.zeros(normalized.shape,dtype=ntl.float32)
    ss=ntl.sum(ntl.where((column//64)==0,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==1,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==2,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==3,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==4,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==5,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==6,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==7,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==8,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==9,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==10,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==11,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==12,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==13,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==14,partial,0.0),-1)
    ss=ss+ntl.sum(ntl.where((column//64)==15,partial,0.0),-1)
    inv=ntl.rsqrt(ss/5376+1e-5)
    sums=ss[:,None]
    inverse=inv[:,None]
    x=hidden.to(ntl.float32)
    normalized=(x*weight.to(ntl.float32))*inv[:,None]

