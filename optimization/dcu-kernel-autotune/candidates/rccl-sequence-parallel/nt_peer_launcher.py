"""独立微测用九齿镜像发射器；通信与发布依赖由transport库执行。"""
import ctypes as ct
import hashlib
import json
from pathlib import Path


def install_adapter(transport,build):
    build=Path(build);meta=json.loads((build/'peer-gate-build.json').read_text())
    assert hashlib.sha256((build/'peer-gate.hsaco').read_bytes()).hexdigest()==meta['hsaco_sha256']
    hip=ct.CDLL('libamdhip64.so')
    hip.hipSetDevice.argtypes=[ct.c_int];hip.hipGetDevice.argtypes=[ct.POINTER(ct.c_int)]
    hip.hipModuleLoad.argtypes=[ct.POINTER(ct.c_void_p),ct.c_char_p]
    hip.hipModuleGetFunction.argtypes=[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p]
    hip.hipModuleLaunchKernel.argtypes=[ct.c_void_p]+[ct.c_uint]*7+[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.c_void_p]
    for f in ['hipSetDevice','hipGetDevice','hipModuleLoad','hipModuleGetFunction','hipModuleLaunchKernel']:getattr(hip,f).restype=ct.c_int
    def check(code):
        if code:raise RuntimeError(f'HIP status {code}')
    modules=[];functions=[];prior=ct.c_int();check(hip.hipGetDevice(ct.byref(prior)))
    for rank in range(8):
        check(hip.hipSetDevice(rank));module=ct.c_void_p();function=ct.c_void_p()
        check(hip.hipModuleLoad(ct.byref(module),str(build/'peer-gate.hsaco').encode()));check(hip.hipModuleGetFunction(ct.byref(function),module,b'h3_peer_gate_kernel'))
        modules.append(module);functions.append(function)
    check(hip.hipSetDevice(prior.value))
    ptr=ct.POINTER(ct.c_uint64);transport.transport_publish.argtypes=[ptr,ptr,ptr,ct.c_size_t,ct.c_int];transport.transport_publish.restype=ct.c_int
    def unsupported(*args):raise RuntimeError('九齿适配器只用于分片门控微测')
    def execute(inputs,parts,outputs,streams,maps,residuals,tables,indices,count,vec,blocks):
        prior=ct.c_int();check(hip.hipGetDevice(ct.byref(prior)))
        for rank in range(8):
            values=[]
            for arg in meta['arguments']:
                source=arg['source']
                if arg['kind']=='tensor':
                    if source.startswith('p') and source[1:].isdigit():value=inputs[int(source[1:])]
                    else:value={'owners':maps[rank],'residual':residuals[rank],'gate':tables[rank],'indices':indices[rank],'output':parts[rank]}[source]
                elif arg['kind']=='scalar':value=rank if source=='rank' else sum(x<<(3*i) for i,x in enumerate([0,1,3,2,4,6,5,7]))
                elif arg['kind']=='shape':value=(count//8 if source in ('owners','output') else count//5376 if source=='indices' else 3 if source=='gate' else count)
                else:raise ValueError(arg)
                values.append(value)
            storage=(ct.c_uint64*(len(values)+2))(*values,0,0)
            params=(ct.c_void_p*(len(values)+2))(*(ct.addressof(storage)+i*8 for i in range(len(values)+2)))
            check(hip.hipSetDevice(rank));check(hip.hipModuleLaunchKernel(functions[rank],(count//8+meta['tile']-1)//meta['tile'],1,1,meta['warps']*64,1,1,meta['shared'],streams[rank],params,None))
        check(transport.transport_publish(parts,outputs,streams,count,blocks));check(hip.hipSetDevice(prior.value));return 0
    transport.peer_setup=transport.transport_setup;transport.peer_cleanup=transport.transport_cleanup
    transport.peer_execute=unsupported;transport.peer_execute_mode=unsupported;transport.peer_execute_gate=execute
    transport._nt_modules=modules;transport._nt_functions=functions
