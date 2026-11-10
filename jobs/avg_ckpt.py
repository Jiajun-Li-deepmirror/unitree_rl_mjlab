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
print(f"wrote {out}: actor averaged over {len(cks)} ({paths[0]} .. {paths[-1]}), iter {avg['iter']}")
