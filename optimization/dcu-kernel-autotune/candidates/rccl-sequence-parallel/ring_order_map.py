"""RCCL Ring/Simple 分块假设的纯索引模型，必须独立验证后才能用于实验。"""
import numpy as np
RING=(0,1,3,2,4,6,5,7)


def owners(indices,count,channels=16,chunk=524288,alignment=1024):
    assert count>0 and count%(channels*8)==0 and channels>0 and chunk>0 and alignment>0
    indices=np.asarray(indices,dtype=np.int64)
    local=indices%(count//channels)
    loopbase=(local//(8*chunk))*(8*chunk)
    remaining=np.minimum(count//channels-loopbase,8*chunk)
    step=(((remaining+7)//8+alignment-1)//alignment)*alignment
    result=(local-loopbase)//step
    assert np.all((result>=0)&(result<8))
    return result.astype(np.uint8)


def table(count,**kwargs):
    # 当前实验的所有源布局、分块边界都要求8个BF16元素对齐。
    return owners(np.arange(count//8,dtype=np.int64)*8,count,**kwargs)
