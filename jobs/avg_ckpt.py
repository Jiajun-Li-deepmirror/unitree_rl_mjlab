"""Average the actor weights (and its observation normalizer) of several checkpoints of one run (2026-10-30: do
nearby checkpoints averaged pass more steadily than any one of them?). The other entries come from the last
checkpoint. Usage: python avg_ckpt.py OUT CKPT ..."""
import sys

import torch

out, paths = sys.argv[1], sys.argv[2:]
cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
avg = cks[-1]
sd = avg["actor_state_dict"]
for key, value in sd.items():
  if torch.is_floating_point(value):
    sd[key] = torch.stack([c["actor_state_dict"][key] for c in cks]).mean(0)
torch.save(avg, out)
# The source run's environment beside the average (scripts/train.py run_env.sh; 2026-11-11), so the average can be
# checked, recorded and exported with the settings it trained with.
import os
import shutil

src_env = os.path.join(os.path.dirname(paths[-1]), "run_env.sh")
if os.path.isfile(src_env):
  shutil.copy(src_env, os.path.join(os.path.dirname(out), "run_env.sh"))
print(f"wrote {out}: actor averaged over {len(cks)} ({paths[0]} .. {paths[-1]}), iter {avg['iter']}")
