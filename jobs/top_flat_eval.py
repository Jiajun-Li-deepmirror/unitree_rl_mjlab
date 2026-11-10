"""top_flat_eval.py (go2-depth 2026-11-04 ~17:00): ws_stair_eval.py plus the tinynav top-landing fault measured on the
training terrain. tinynav's stair_wheels.py counts front wheels lifted (> 0.06 s, apex > 2 cm) on map3's top landing, 0-40
cm past the last riser while the rear wheels still climb: As2W students 3.2-4.0 per climb, with or without wheelahead.
ws_stair_eval's "top landing" zone (even within r + 0.15 all round, higher than the base) also holds wheels in the
middle of deep treads, where lifts are stair steps. Here the zone "top_flat" is: a drivable command, the robot's scan
circle uneven, the wheel's ground > 5 cm above the base's, and no height change > 5 cm within 0.35 m AHEAD of the
wheel (wheel_uneven_mask with ahead_command; the riser just climbed behind it ignored). Reported: the share of wheel
time lifted there, and lift events (take-off in the zone, airborne > 0.06 s, apex > 2 cm) per 100 zone wheel-seconds.

ws_stair_eval.py (go2-depth 2026-11-03): the wheelstep round's two faults, measured for a wheeled teacher checkpoint
on its training terrain at fixed levels (no terrain curriculum: every robot keeps a level drawn uniformly over all rows,
the same draw for every checkpoint given the seed), so an old and a new teacher compare on equal ground.

  BLIND_PILOT=<pilot>,wheelstep python jobs/ws_stair_eval.py --task Unitree-Go2W-Depth-Polish CKPT [--num-envs 512]

Per robot and step (control dt) it counts:
- lift_even: wheels off the ground under a drivable command (moving, |vy| <= 0.1, |wz| <= 0.1, not marching) with no
  riser within r + 0.15 m (mdp.wheel_uneven_mask), over wheels on such ground under such commands (fault 1);
- nosing: wheels hanging on a nosing (mdp.wheel_nosing), per 1000 wheel-steps on stairs robots (fault 2);
- clean: of the first touchdowns after a riser trigger (wheel_riser_lift's tries), the share that are clean landings
  on the riser top (fault 2);
plus falls (terminations other than time-outs) per 1000 robot-seconds as a sanity line.
The deterministic actor (mean action) of a depth teacher checkpoint (rsl_rl MLP with obs normalizer).
"""

import argparse

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src.tasks.velocity import mdp
from src.tasks.velocity.blind_env_cfg import FOOT_SITE_NAMES

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("checkpoint")
p.add_argument("--num-envs", type=int, default=512)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--runner", action="store_true", help="load through the task's runner (depth students; run via student_eval.py)")
# 2026-11-05: the Scratch tasks' "stairs" terrain is pyramids (Go2W teacher slid back 14.8% there, 3.3% on the Polish
# mix); tinynav's map3 L0 is a walled straight flight. --only walled_up --difficulty 0.7 1.0 evaluates on those alone.
p.add_argument("--only", default=None, help="keep only this sub-terrain (e.g. walled_up)")
p.add_argument("--difficulty", type=float, nargs=2, default=None, help="the generator's difficulty range (lo hi)")
args = p.parse_args()

cfg = load_env_cfg(args.task)
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
assert "wheel_riser_lift" in cfg.rewards, "a wheeled stairs task"
cfg.curriculum.pop("terrain_levels", None)
gen = cfg.scene.terrain.terrain_generator
if args.only is not None:
  from dataclasses import replace as _replace
  assert args.only in gen.sub_terrains, (args.only, list(gen.sub_terrains))
  gen.sub_terrains = {args.only: _replace(gen.sub_terrains[args.only], proportion=1.0)}
if args.difficulty is not None:
  gen.difficulty_range = tuple(args.difficulty)
if args.only is not None:
  # all-walled tiles: the robot's default pose (before the first reset) sits in a wall block (B2W 122 contacts)
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 160)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 800)
  cfg.curriculum.pop("stairs_share", None)  # asserts stair and other columns
cfg.scene.terrain.max_init_terrain_level = gen.num_rows - 1

counts = {"tries": 0.0, "clean": 0.0, "apex": []}


class CountingLift(mdp.wheel_riser_lift):
  """wheel_riser_lift that also counts its landing tries and clean landings (the same tests, before the update)."""

  def __call__(self, env, **kw):
    contact = env.scene[kw["sensor_name"]]
    asset = env.scene[kw["asset_cfg"].name]
    in_contact = contact.data.found > 0
    force = contact.data.force
    fxy, fz = torch.norm(force[..., :2], dim=-1), torch.abs(force[..., 2])
    stance = in_contact & (fz > torch.clamp(kw.get("stance_ratio", 2.0) * fxy, min=kw.get("stance_min_force", 10.0)))
    sole_z = asset.data.site_pos_w[:, kw["asset_cfg"].site_ids, 2] - kw["wheel_radius"]
    active = ~mdp.rewards._zero_command_mask(env, kw["command_name"], kw["command_threshold"])
    tried = in_contact & ~self.prev_contact & ~self.land_paid & (self.time_left > 0.0) & active[:, None]
    clean = tried & stance & (sole_z >= self.riser_top - kw.get("land_tolerance", 0.02))
    counts["tries"] += float(tried.sum())
    counts["clean"] += float(clean.sum())
    # the sole's apex over the riser top during the lift before each try (m)
    if not hasattr(self, "apex"):
      self.apex = torch.full_like(sole_z, -torch.inf)
    self.apex = torch.where(~in_contact, torch.maximum(self.apex, sole_z), self.apex)
    if tried.any():
      counts["apex"].append((self.apex - self.riser_top)[tried].cpu())
    self.apex = torch.where(in_contact, torch.full_like(sole_z, -torch.inf), self.apex)
    return super().__call__(env, **kw)


cfg.rewards["wheel_riser_lift"].func = CountingLift
_lift = cfg.rewards["wheel_riser_lift"].params
r = _lift["wheel_radius"]
# the metrics' own params (not the task's reward terms: Scratch tasks have no wheelstep / noslide terms)
nose_params = {"sensor_name": _lift["sensor_name"], "scan_sensor_name": _lift["scan_sensor_name"], "wheel_radius": r}
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
agent_cfg = load_rl_cfg(args.task)
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
sites = SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)
sites.resolve(env.scene)
from mjlab.managers.reward_manager import RewardTermCfg as _RTC  # noqa: E402

slide_term = mdp.wheel_slide_back(_RTC(func=mdp.wheel_slide_back, weight=1.0, params={"asset_cfg": sites}), env)

if args.runner:
  from dataclasses import asdict

  from mjlab.tasks.registry import load_runner_cls

  runner = load_runner_cls(args.task)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  _policy = runner.get_inference_policy(device="cuda:0")
else:
  ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  sd = {k: v.to("cuda:0") for k, v in ck["actor_state_dict"].items()}
  layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})


def act(obs):
  return _policy(obs) if args.runner else actor(obs["actor"])


def actor(x):
  x = (x - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 1e-2)
  for i, n in enumerate(layers):
    x = x @ sd[f"mlp.{n}.weight"].T + sd[f"mlp.{n}.bias"]
    if i < len(layers) - 1:
      x = torch.nn.functional.elu(x)
  return x


cmd_term = env.command_manager.get_term("twist")
tf = {"air": None}
acc = {"tf_num": 0.0, "tf_den": 0.0, "tf_ev": 0.0, "top_num": 0.0, "top_den": 0.0, "bot_num": 0.0, "bot_den": 0.0, "slide": 0.0, "stub": 0.0, "w2": 0.0, "wpk": [], "st_n": 0.0, "near_num": 0.0, "near_den": 0.0, "lift_num": 0.0, "lift_den": 0.0, "nose": 0.0, "nose_den": 0.0, "falls": 0.0}
with torch.inference_mode():
  env.reset()
  obs = venv.get_observations()
  for step in range(args.warmup + args.steps):
    if step == args.warmup:
      counts["tries"] = counts["clean"] = 0.0
      counts["apex"] = []
    obs, _, dones, extras = venv.step(act(obs))
    if step < args.warmup:
      continue
    contact = env.scene["feet_ground_contact"]
    in_air = contact.data.found <= 0
    cmd = env.command_manager.get_command("twist")
    drivable = (~mdp.rewards._zero_command_mask(env, "twist", 0.1)) & (cmd[:, 1].abs() <= 0.1) & (cmd[:, 2].abs() <= 0.1)
    march = getattr(cmd_term, "march", None)
    if march is not None:
      drivable &= ~march
    even = ~mdp.wheel_uneven_mask(env, "terrain_scan", sites, r + 0.15)
    ok = even & drivable[:, None]
    acc["lift_num"] += float((in_air & ok).sum())
    acc["lift_den"] += float(ok.sum())
    # fault 1 itself: this wheel on even ground while the robot's 0.45 m circle holds a riser (the old free zone)
    near = ok & mdp.uneven_terrain_mask(env, "terrain_scan")[:, None]
    acc["near_num"] += float((in_air & near).sum())
    # the near zone split by where the wheel is: on ground higher than under the base (front wheels on a top landing)
    # or lower (rear wheels on the start landing); terrain heights from the scan (max within r of the wheel, median
    # within 0.2 m of the base)
    scan = env.scene["terrain_scan"].data
    hit = scan.hit_pos_w
    wxy = env.scene["robot"].data.site_pos_w[:, sites.site_ids, :2]
    dw = torch.norm(hit[:, None, :, :2] - wxy[:, :, None, :], dim=-1)
    wz = torch.where((dw <= r) & (scan.distances >= 0)[:, None, :], hit[:, None, :, 2], -torch.inf).amax(dim=-1)
    db = torch.norm(hit[..., :2] - scan.pos_w[:, None, :2], dim=-1)
    bz = torch.where((db <= 0.2) & (scan.distances >= 0), hit[..., 2], torch.nan).nanmedian(dim=1).values
    higher = wz > bz[:, None] + 0.05
    lower = wz < bz[:, None] - 0.05
    acc["top_num"] += float((in_air & near & higher).sum()); acc["top_den"] += float((near & higher).sum())
    acc["bot_num"] += float((in_air & near & lower).sum()); acc["bot_den"] += float((near & lower).sum())
    acc["near_den"] += float(near.sum())
    # top_flat (see the docstring)
    ahead_even = ~mdp.wheel_uneven_mask(env, "terrain_scan", sites, 0.35, ahead_command="twist")
    zone = drivable[:, None] & ahead_even & mdp.uneven_terrain_mask(env, "terrain_scan")[:, None] & higher
    acc["tf_num"] += float((in_air & zone).sum()); acc["tf_den"] += float(zone.sum())
    wz3 = env.scene["robot"].data.site_pos_w[:, sites.site_ids, 2]
    if tf["air"] is None:
      tf["air"] = torch.zeros_like(wz3); tf["z0"] = wz3.clone(); tf["apex"] = wz3.clone()
      tf["in"] = torch.zeros_like(wz3, dtype=torch.bool); tf["prev"] = ~in_air
    off = in_air & tf["prev"]  # take-off this step
    tf["in"] = torch.where(off, zone, tf["in"]); tf["z0"] = torch.where(off, wz3, tf["z0"])
    tf["apex"] = torch.where(off, wz3, torch.maximum(tf["apex"], wz3))
    tf["air"] = torch.where(in_air, tf["air"] + env.step_dt, tf["air"])
    land = ~in_air & ~tf["prev"]
    ev = land & tf["in"] & (tf["air"] > 0.06) & (tf["apex"] - tf["z0"] > 0.02)
    acc["tf_ev"] += float(ev.sum())
    tf["air"] = torch.where(~in_air, 0.0, tf["air"]); tf["in"] = tf["in"] & in_air
    tf["prev"] = ~in_air
    if dones.any():
      d = dones.bool()
      tf["air"][d] = 0.0; tf["in"][d] = False
    stairs = cmd_term.is_stair_env
    # body stability on the stairs (robots on stairs columns over uneven ground): pitch / roll rate
    on_st = stairs & mdp.uneven_terrain_mask(env, "terrain_scan")
    wxy = env.scene["robot"].data.root_link_ang_vel_b[:, :2]
    acc["w2"] += float((wxy.square().sum(1) * on_st.float()).sum())
    acc["st_n"] += float(on_st.sum())
    if on_st.any():
      acc["wpk"].append(wxy[on_st].abs().amax(1).cpu())
    nose = mdp.wheel_nosing(env, **{**nose_params, "asset_cfg": sites, "clearance": 0.02, "reach": 0.05})
    acc["slide"] += float(slide_term(env, sensor_name=_lift["sensor_name"], wheel_radius=r, command_name="twist",
                                     command_threshold=0.1, asset_cfg=sites).sum()) * env.step_dt
    if dones.any():
      slide_term.reset(torch.nonzero(dones.bool()).squeeze(-1))
    acc["nose"] += float((nose * stairs.float()).sum())
    # any wheel pushing on a riser it has not cleared (tilted contact, a tread top within r + 0.1 m above the sole by
    # more than r / 2, at any height: nosings and riser faces / corners)
    stub = mdp.wheel_nosing(env, **{**nose_params, "asset_cfg": sites, "clearance": 10.0, "reach": 0.1})
    acc["stub"] += float((stub * stairs.float()).sum())
    acc["nose_den"] += float(stairs.sum()) * len(FOOT_SITE_NAMES)
    timeouts = extras.get("time_outs", torch.zeros_like(dones))
    acc["falls"] += float((dones.bool() & ~timeouts.bool()).sum())
env.close()
secs = args.steps * env.step_dt * args.num_envs
print(f"top_flat_eval {args.checkpoint}: top_flat lifted {100 * acc['tf_num'] / max(acc['tf_den'], 1):.2f}% of "
      f"{acc['tf_den'] * env.step_dt:.0f} wheel-s, {100 * acc['tf_ev'] / max(acc['tf_den'] * env.step_dt, 1e-9):.1f} lifts per 100 wheel-s ({int(acc['tf_ev'])})", flush=True)
print(f"ws_stair_eval {args.checkpoint}: lift_near {100 * acc['near_num'] / max(acc['near_den'], 1):.2f}% "
      f"(top landing {100 * acc['top_num'] / max(acc['top_den'], 1):.2f}%, start landing {100 * acc['bot_num'] / max(acc['bot_den'], 1):.2f}%) | lift_even {100 * acc['lift_num'] / max(acc['lift_den'], 1):.2f}% | "
      f"nosing {1000 * acc['nose'] / max(acc['nose_den'], 1):.2f} per 1000 stairs wheel-steps | riser contact {1000 * acc['stub'] / max(acc['nose_den'], 1):.2f} | "
      f"clean {100 * counts['clean'] / max(counts['tries'], 1):.1f}% of {int(counts['tries'])} tries | "
      f"slide back {100 * acc['slide'] / max(counts['tries'], 1):.1f}% of tries ({int(acc['slide'])}) | "
      + (lambda a: f"apex over riser top p10/p50/p90 {100 * float(torch.quantile(a, 0.1)):+.1f}/{100 * float(torch.quantile(a, 0.5)):+.1f}/{100 * float(torch.quantile(a, 0.9)):+.1f} cm | ")(torch.cat(counts["apex"]).clamp(-1, 1) if counts["apex"] else torch.zeros(1))
      + f"falls {1000 * acc['falls'] / secs:.2f} per 1000 robot-s | stairs ang vel xy RMS {(acc['w2'] / max(acc['st_n'], 1)) ** 0.5:.2f} rad/s, "
      f"p99 {float(torch.quantile(torch.cat(acc['wpk'])[:1000000], 0.99)) if acc['wpk'] else float('nan'):.2f} rad/s")
