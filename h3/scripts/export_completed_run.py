import time,json,shutil,hashlib
from pathlib import Path
import av
from PIL import Image,ImageDraw
r=Path.cwd();d=r/'outputs/public_fl2va/20261107_165708_direct_infinilm';dest=Path('/dev/shm/infinifabric-shell/b2a0ae4c4cfb48c2af1e8c18b0503f77/workspace')
for _ in range(150):
 report=json.loads((d/'run.json').read_text())
 if report['phase'] in ('complete','failed'):break
 time.sleep(20)
else:raise TimeoutError('generation still running')
if not report.get('generation_succeeded'):raise RuntimeError(report)
sheet=Image.new('RGB',(1344,560),'white');draw=ImageDraw.Draw(sheet);picked=[]
with av.open(str(d/'video.mp4')) as c:
 for i,f in enumerate(c.decode(video=0)):
  if i in (0,24,48,72,96,123):
   k=len(picked);sheet.paste(f.to_image().resize((448,256)),((k%3)*448,(k//3)*280));draw.text(((k%3)*448+5,(k//3)*280+258),'frame '+str(i),fill='black');picked.append(i)
sheet.save(d/'contact-sheet.jpg',quality=92)
for name in ('video.mp4','run.json','generation.log','contact-sheet.jpg'):shutil.copyfile(d/name,dest/('h3-'+name))
print('EXPORT_COMPLETE',dest,'FRAMES',picked,'VIDEO_SHA256',hashlib.sha256((d/'video.mp4').read_bytes()).hexdigest(),flush=True)
