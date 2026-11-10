"""avg_student.py OUT CKPT ... (depth_fe, 2026-10-31): go2-home's avg_ckpt.py for a distilled depth student: the
student's weights (student_state_dict: encoder, GRU, heads and its observation normalizers) averaged over several
checkpoints of one run; everything else (teacher, optimizer, iter) from the last. Integer buffers (counts) come
from the last checkpoint."""
import sys

import torch

out, paths = sys.argv[1], sys.argv[2:]
cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
avg = cks[-1]
sd = avg["student_state_dict"]
n = 0
for key, value in sd.items():
  if torch.is_floating_point(value):
    sd[key] = torch.stack([c["student_state_dict"][key] for c in cks]).mean(0)
    n += 1
torch.save(avg, out)
print(f"wrote {out}: student averaged over {len(cks)} ({paths[0]} .. {paths[-1]}), {n} tensors, iter {avg['iter']}")
