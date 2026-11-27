"""九齿表达的原序BF16分片求和与门控；主机侧提供已校准的位置表和rank环。"""
import ninetoothed.language as ntl


def arrangement(p0,p1,p2,p3,p4,p5,p6,p7,owners,residual,gate,indices,rank,ringbits,output,*,tile=256):
    return p0,p1,p2,p3,p4,p5,p6,p7,owners,residual,gate,indices,rank,ringbits,output.tile((tile,))


def application(p0,p1,p2,p3,p4,p5,p6,p7,owners,residual,gate,indices,rank,ringbits,output):
    local=output.offsets(0).to(ntl.int32)
    index=rank*(p0.shape[0]//8)+local
    owner=owners.source[index//8].to(ntl.int32)
    v0=p0.source[index].to(ntl.float32)
    v1=p1.source[index].to(ntl.float32)
    v2=p2.source[index].to(ntl.float32)
    v3=p3.source[index].to(ntl.float32)
    v4=p4.source[index].to(ntl.float32)
    v5=p5.source[index].to(ntl.float32)
    v6=p6.source[index].to(ntl.float32)
    v7=p7.source[index].to(ntl.float32)
    r=(ringbits>>(3*((owner+1)&7)))&7
    acc=ntl.where(r==0,v0,ntl.where(r==1,v1,ntl.where(r==2,v2,ntl.where(r==3,v3,ntl.where(r==4,v4,ntl.where(r==5,v5,ntl.where(r==6,v6,v7)))))))
    for step in range(2,9):
        r=(ringbits>>(3*((owner+step)&7)))&7
        value=ntl.where(r==0,v0,ntl.where(r==1,v1,ntl.where(r==2,v2,ntl.where(r==3,v3,ntl.where(r==4,v4,ntl.where(r==5,v5,ntl.where(r==6,v6,v7)))))))
        bits=(acc+value).to(ntl.uint32,bitcast=True)
        rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
        acc=rounded.to(ntl.float32,bitcast=True)
    row=index//5376
    column=index%5376
    g=gate.source[indices.source[row].to(ntl.int32),column].to(ntl.float32)
    scaled=(g*acc).to(ntl.bfloat16)
    output=(residual.source[index].to(ntl.float32)+scaled.to(ntl.float32)).to(ntl.bfloat16)
