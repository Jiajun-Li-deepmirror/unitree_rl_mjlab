# map3 stairwell (tinynav-sim)

Copied from tinynav-sim `mujoco/assets/mjcf/` (commit 011364c):

- `map3_collision.xml`: the 123 collision boxes of the five-story map3
  stairwell (flights, landings, floors), generated from the Blender rebuild.
- `scene.xml`: the tinynav-sim scene that includes it, plus the sim guard rails
  (`rail_*`) around the stairwell well and slots.

Used only for evaluation (`scripts/go2_depth/eval_map3.py`); map3 is never part
of the training terrain. Keep in sync with tinynav-sim if the map changes.
