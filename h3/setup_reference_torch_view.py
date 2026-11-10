"""Create a reference-only view of unchanged platform Torch files."""
import os
from pathlib import Path

afs = Path(os.environ["IFAB_AFS_HOME"])
root = afs / ".reference-runtime/python"
root.mkdir(parents=True, exist_ok=True, mode=0o770)
source = Path("/usr/local/lib/python3.11/site-packages/torch")
target = root / "torch"
target.mkdir(exist_ok=True, mode=0o770)
for entry in source.iterdir():
    if entry.name == "lib":
        continue
    link = target / entry.name
    if not link.exists():
        link.symlink_to(entry, target_is_directory=entry.is_dir())
libs = target / "lib"
libs.mkdir(exist_ok=True, mode=0o770)
for entry in (source / "lib").iterdir():
    link = libs / entry.name
    if not link.exists():
        link.symlink_to(entry, target_is_directory=entry.is_dir())
extra = source.parent / "torch.libs"
if extra.exists() and not (root / "torch.libs").exists():
    (root / "torch.libs").symlink_to(extra, target_is_directory=True)
flash = next((afs / ".reference-runtime/flash-attn").glob("flash_attn_2_cuda*.so")).resolve()
for directory in (root, target, libs):
    link = directory / flash.name
    if not link.exists():
        link.symlink_to(flash)
print("REFERENCE_TORCH_VIEW_READY")
