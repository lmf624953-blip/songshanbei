"""九齿V预排布：置换LDS字节位，使原read2st64的两半变成连续16字节。"""

import ninetoothed.language as ntl

from v_prepack_kernel import arrangement


def zipped(value, blocks, output):
    index=output.offsets(0).to(ntl.int32)
    head=index//(blocks*4096)
    tile=(index//4096)%blocks
    local=index%4096
    # 字节bit3/10互换，即BF16元素索引bit2/9互换；置换为自身的逆。
    local=(local&~516)|((local&4)<<7)|((local&512)>>7)
    column=local//32
    pair=(local//2)%16
    row=2*(pair^(((column//2)%8)*2))+local%2
    output=value.source[tile*32+row,head,column]


def bank_zipped(value, blocks, output):
    index=output.offsets(0).to(ntl.int32)
    head=index//(blocks*4096)
    tile=(index//4096)%blocks
    local=index%4096
    # 再换字节bit5/7，使连续lane在ds_read_b128的bank组中展开。
    local=(local&~596)|((local&4)<<7)|((local&512)>>7)|((local&16)<<2)|((local&64)>>2)
    column=local//32
    pair=(local//2)%16
    row=2*(pair^(((column//2)%8)*2))+local%2
    output=value.source[tile*32+row,head,column]
