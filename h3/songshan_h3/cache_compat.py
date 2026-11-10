"""Use same-directory atomic cache writes on the InfiniFabric filesystem."""
import os,uuid
from triton.runtime.cache import FileCacheManager
class SameDirectoryCacheManager(FileCacheManager):
 def put(self,data,filename,binary=True):
  binary=isinstance(data,bytes)
  if not binary:data=str(data)
  path=self._make_path(filename)
  temporary=self._make_path('.tmp-'+str(os.getpid())+'-'+uuid.uuid4().hex)
  try:
   with open(temporary,'wb' if binary else 'w') as f:f.write(data)
   os.replace(temporary,path)
  finally:
   if os.path.exists(temporary):os.unlink(temporary)
  return path
