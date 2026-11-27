import ctypes as c,json,os,sys
h=c.CDLL('libamdhip64.so');v=c.CDLL(sys.argv[1]);streams=(c.c_uint64*8)()
h.hipStreamCreate.argtypes=[c.POINTER(c.c_void_p)];h.hipSetDevice.argtypes=[c.c_int];h.hipMemGetInfo.argtypes=[c.POINTER(c.c_size_t)]*2
for r in range(8):
 assert h.hipSetDevice(r)==0
 s=c.c_void_p();assert h.hipStreamCreate(c.byref(s))==0;streams[r]=s.value
v.vm_setup.argtypes=[c.c_size_t,c.POINTER(c.c_uint64),c.c_void_p,c.POINTER(c.c_uint64),c.POINTER(c.c_size_t),c.POINTER(c.c_size_t)];v.vm_setup.restype=c.c_char_p;v.vm_close.restype=c.c_char_p
class Location(c.Structure):_fields_=[('type',c.c_int),('id',c.c_int)]
class Flags(c.Structure):_fields_=[('compression',c.c_ubyte),('rdma',c.c_ubyte),('usage',c.c_ushort)]
class Prop(c.Structure):_fields_=[('type',c.c_int),('handles',c.c_int),('location',Location),('metadata',c.c_void_p),('flags',Flags)]
assert c.sizeof(Prop)==32
h.hipMemRetainAllocationHandle.argtypes=[c.POINTER(c.c_uint64),c.c_void_p];h.hipMemGetAllocationPropertiesFromHandle.argtypes=[c.POINTER(Prop),c.c_uint64];h.hipMemRelease.argtypes=[c.c_uint64]
def free():
 result=[]
 for r in range(8):
  assert h.hipSetDevice(r)==0;assert h.hipDeviceSynchronize()==0
  available=c.c_size_t();total=c.c_size_t();assert h.hipMemGetInfo(c.byref(available),c.byref(total))==0;result.append(available.value)
 return result
records=[]
for rows,gran in [(1024,4096),(27570,2097152)]:
 os.environ['H3_VMM_MAPPING_GRANULARITY']=str(gran);before=free();address=c.c_uint64();bounds=(c.c_size_t*9)();g=c.c_size_t();error=v.vm_setup(rows,streams,None,c.byref(address),bounds,c.byref(g));assert not error,error
 after=free();chunk=(((rows*5376*2+7)//8+gran-1)//gran)*gran;props=[]
 for r in range(8):
  assert h.hipSetDevice(r)==0;handle=c.c_uint64();status=h.hipMemRetainAllocationHandle(c.byref(handle),address.value+r*chunk);prop=Prop()
  propstatus=h.hipMemGetAllocationPropertiesFromHandle(c.byref(prop),handle.value) if status==0 else None
  props.append(dict(requested_owner=r,retain_status=status,properties_status=propstatus,returned_location_type=prop.location.type,returned_location_id=prop.location.id))
  if status==0:assert h.hipMemRelease(handle.value)==0
 error=v.vm_close();assert not error,error
 records.append(dict(rows=rows,granularity=gran,physical_chunk_bytes=chunk,free_before=before,free_after_setup=after,used_delta_bytes=[a-b for a,b in zip(before,after)],properties=props,free_after_close=free()))
print(json.dumps(dict(scope='独立物理归属/显存记账诊断，无模型或性能结论',records=records),ensure_ascii=False,indent=2))
