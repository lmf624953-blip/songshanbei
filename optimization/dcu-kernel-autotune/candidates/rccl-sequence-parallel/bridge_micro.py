"""在独立八线程微测中使用与模型相同的InfiniCCL communicator和主机桥。"""
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path


class InfiniCCLProxy:
    def __init__(self,path):
        self.lib=ct.CDLL(str(Path(path).resolve()));p=ct.c_void_p
        signatures={'infinicclCommInitAll':[ct.c_int,ct.POINTER(p),ct.c_int,ct.POINTER(ct.c_int)],'infinicclCommDestroy':[p],
                    'infinicclGroupStart':[p],'infinicclGroupEnd':[p],
                    'infinicclAllReduce':[p,p,ct.c_size_t,ct.c_int,ct.c_int,p,p]}
        for name,args in signatures.items():getattr(self.lib,name).argtypes=args;getattr(self.lib,name).restype=ct.c_int
        self.comms=None
    @staticmethod
    def check(code):
        if code:raise RuntimeError(f'InfiniCCL status {code}')
    def ncclCommInitAll(self,comms,count,devices):
        # 实际安装infinicore.h：HYGON=8、BF16=19。
        self.check(self.lib.infinicclCommInitAll(8,comms,count,devices));self.comms=comms
    def ncclCommDestroy(self,comm):self.check(self.lib.infinicclCommDestroy(comm))
    def ncclGroupStart(self):self.check(self.lib.infinicclGroupStart(self.comms[0]))
    def ncclGroupEnd(self):self.check(self.lib.infinicclGroupEnd(self.comms[0]))
    def ncclAllReduce(self,src,dst,count,dtype,op,comm,stream):
        assert dtype==9 and op==0
        self.check(self.lib.infinicclAllReduce(src,dst,count,19,0,comm,stream))


def install_adapter(peer,build,indexed,proxy):
    build=Path(build);indexed=Path(indexed);meta=json.loads((build/'peer-runtime-build.json').read_text())
    native=ct.CDLL(str(indexed/'indexed-host.so'))
    peer.h3_peer_bridge_prepare.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.POINTER(ct.c_int),ct.c_int,ct.c_char_p,ct.c_void_p];peer.h3_peer_bridge_prepare.restype=ct.c_char_p
    peer.h3_peer_bridge_raw.argtypes=[ct.c_int]+[ct.c_uint64]*5+[ct.c_size_t,ct.c_int,ct.c_uint64,ct.c_uint64];peer.h3_peer_bridge_raw.restype=ct.c_char_p
    peer.h3_peer_bridge_stats.restype=ct.c_char_p
    codes=(ct.c_int*len(meta['abi_codes']))(*meta['abi_codes']);pool=ThreadPoolExecutor(max_workers=8)
    def prepare():
        message=peer.h3_peer_bridge_prepare(str(build/'peer-gate.hsaco').encode(),meta['nt']['tile'],meta['nt']['warps'],codes,len(codes),str(build/'peer-transport.so').encode(),ct.cast(native.songshan_bench_indexed,ct.c_void_p))
        if message:raise RuntimeError(message.decode())
        return 0
    def unsupported(*args):raise RuntimeError('主机桥只支持分片门控微测')
    def execute(inputs,parts,outputs,streams,maps,residuals,tables,indices,count,vec,blocks):
        def call(rank):
            msg=peer.h3_peer_bridge_raw(rank,inputs[rank],outputs[rank],residuals[rank],tables[rank],indices[rank],count,3,proxy.comms[rank],streams[rank])
            if msg:raise RuntimeError(msg.decode())
        futures=[pool.submit(call,r) for r in range(8)]
        for f in futures:f.result()
        return 0
    def cleanup():pool.shutdown(wait=True);return 0
    peer.peer_setup=prepare;peer.peer_cleanup=cleanup;peer.peer_execute=unsupported;peer.peer_execute_mode=unsupported;peer.peer_execute_gate=execute
