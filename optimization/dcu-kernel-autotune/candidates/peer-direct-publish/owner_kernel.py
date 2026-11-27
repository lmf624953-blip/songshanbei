"""按已校准RCCL owner分工；主机预排8个输入指针，保留逐步BF16累加顺序。"""
import ninetoothed.language as ntl


def arrangement(p0,p1,p2,p3,p4,p5,p6,p7,residual,gate,indices,rank,out0,out1,out2,out3,out4,out5,out6,out7,*,tile=1024):
    length=p0.shape[0]//16
    full=length//4194304
    remaining=length-full*4194304
    capacity=full*524288+((remaining+7)//8+1023)//1024*1024
    return p0,p1,p2,p3,p4,p5,p6,p7,residual,gate,indices,rank,out0[:,:capacity].tile((1,tile)),out1[:,:capacity].tile((1,tile)),out2[:,:capacity].tile((1,tile)),out3[:,:capacity].tile((1,tile)),out4[:,:capacity].tile((1,tile)),out5[:,:capacity].tile((1,tile)),out6[:,:capacity].tile((1,tile)),out7[:,:capacity].tile((1,tile))


def application(p0,p1,p2,p3,p4,p5,p6,p7,residual,gate,indices,rank,out0,out1,out2,out3,out4,out5,out6,out7):
    per_channel=p0.shape[0]//16
    full=per_channel//4194304
    remaining=per_channel-full*4194304
    tail_step=((remaining+7)//8+1023)//1024*1024
    channel=out0.offsets(0).to(ntl.int32)
    local=out0.offsets(1).to(ntl.int32)
    iteration=local//524288
    within=local%524288
    step=ntl.where(iteration<full,524288,tail_step)
    valid=(iteration<full)|((iteration==full)&(within<tail_step))
    position=ntl.where(valid,iteration*4194304+rank*step+within,per_channel)
    index=ntl.where(position<per_channel,channel*per_channel+position,p0.shape[0])
    acc=p0.source[index].to(ntl.float32)
    value=p1.source[index].to(ntl.float32)
    bits=(acc+value).to(ntl.uint32,bitcast=True)
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    acc=rounded.to(ntl.float32,bitcast=True)
    value=p2.source[index].to(ntl.float32)
    bits=(acc+value).to(ntl.uint32,bitcast=True)
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    acc=rounded.to(ntl.float32,bitcast=True)
    value=p3.source[index].to(ntl.float32)
    bits=(acc+value).to(ntl.uint32,bitcast=True)
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    acc=rounded.to(ntl.float32,bitcast=True)
    value=p4.source[index].to(ntl.float32)
    bits=(acc+value).to(ntl.uint32,bitcast=True)
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    acc=rounded.to(ntl.float32,bitcast=True)
    value=p5.source[index].to(ntl.float32)
    bits=(acc+value).to(ntl.uint32,bitcast=True)
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    acc=rounded.to(ntl.float32,bitcast=True)
    value=p6.source[index].to(ntl.float32)
    bits=(acc+value).to(ntl.uint32,bitcast=True)
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    acc=rounded.to(ntl.float32,bitcast=True)
    value=p7.source[index].to(ntl.float32)
    bits=(acc+value).to(ntl.uint32,bitcast=True)
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    acc=rounded.to(ntl.float32,bitcast=True)
    row=index//5376
    column=index%5376
    g=gate.source[indices.source[row].to(ntl.int32),column].to(ntl.float32)
    scaled=(g*acc).to(ntl.bfloat16)
    result=(residual.source[index].to(ntl.float32)+scaled.to(ntl.float32)).to(ntl.bfloat16)
    out0.source[channel,position]=result
    out1.source[channel,position]=result
    out2.source[channel,position]=result
    out3.source[channel,position]=result
    out4.source[channel,position]=result
    out5.source[channel,position]=result
    out6.source[channel,position]=result
    out7.source[channel,position]=result
