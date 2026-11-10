import os,sys,json,time,subprocess,traceback
from pathlib import Path
r=Path(os.environ['IFAB_AFS_HOME']);sys.path.insert(0,str(r))
info=json.loads((r/'artifacts/official-fl2va-command.json').read_text())
assert info['attention_all_8_passed']
import torch
assert torch.cuda.device_count()==8
assert all(torch.cuda.get_device_properties(i).gcnArchName.startswith('gfx936') for i in range(8))
d=r/'outputs/public_fl2va'/(time.strftime('%Y%m%d_%H%M%S')+'_direct_infinilm');d.mkdir(parents=True)
cmd=[('--output='+str(d/'video.mp4')) if x.startswith('--output=') else x for x in info['command']]
report=dict(command=cmd,phase='generation',generation_succeeded=False,content_validation_passed=None,competition_validation_passed=False,attention_validation='artifacts/attention-bridge-device.json')
def save(): (d/'run.json').write_text(json.dumps(report,indent=2))
save();print('DIRECT_RUN',d,flush=True);start=time.monotonic()
try:
 with (d/'generation.log').open('w') as log: ret=subprocess.run(cmd,cwd=info['cwd'],stdout=log,stderr=subprocess.STDOUT)
 report.update(returncode=ret.returncode,wall_seconds=time.monotonic()-start)
 assert ret.returncode==0,ret.returncode
 from songshan_h3.output_validation import validate_generated_media
 report.update(validate_generated_media(d/'video.mp4',dict(num_frames=124,height=768,width=1344,fps=24)))
 report['generation_succeeded']=report['video_specs_passed'];report['phase']='complete'
except BaseException:
 report.update(phase='failed',error=traceback.format_exc());print(report['error'],flush=True)
finally: save();print(json.dumps(report,indent=2),flush=True)
sys.exit(0 if report['generation_succeeded'] else 1)
