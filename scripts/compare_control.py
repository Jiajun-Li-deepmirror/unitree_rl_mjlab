"""Compare local control diagnostics at matched original-reference times."""
import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import PROJECT_ROOT, project_relative


def predicted_pd_force(data):
  ids = [data['joint_names'].tolist().index(n) for n in data['actuator_joint_names']]
  force = ((data['actuator_ctrl'] - data['joint_pos'][:, ids])*data['actuator_kp']
           - data['joint_vel'][:, ids]*data['actuator_kd'])
  return np.clip(force, -data['actuator_force_limits'], data['actuator_force_limits'])


def first_upcross(t, values, threshold, held=False):
  crossings = np.flatnonzero((values[:-1] < threshold) & (values[1:] >= threshold))
  if not len(crossings):
    return None
  i = crossings[0]
  if held:
    return float(t[i+1])
  return float(t[i] + (threshold-values[i])/(values[i+1]-values[i])*(t[i+1]-t[i]))


def load(prefix):
  p = Path(prefix)
  meta = json.loads(Path(str(p)+'.json').read_text())
  with np.load(Path(str(p)+'.npz'), allow_pickle=False) as f:
    control = dict(f)
  with np.load(p.with_name(p.name+'_substeps.npz'), allow_pickle=False) as f:
    sub = dict(f)
  with np.load(p.with_name(p.name+'_reference_kinematics.npz'), allow_pickle=False) as f:
    kin = dict(f)
  return meta, control, sub, kin


def analyze(meta, c, s, kin, focus):
  t, ts = c['reference_t'], s['reference_t']
  mask = (t >= focus[0]-1e-8) & (t <= focus[1]+1e-8)
  if not mask.any() or t[-1] < focus[1]-1e-8:
    return {'completed_diagnostic_window': False, 'termination': meta['termination']}
  frames = np.flatnonzero(mask)
  foot = c['body_names'].tolist().index('left_ankle_roll_link')
  knee = c['joint_names'].tolist().index('left_knee_joint')
  refh, actualh = c['reference_body_pos_world'][:, foot, 2], c['policy_body_pos_world'][:, foot, 2]
  peak = frames[np.argmax(refh[mask])]
  actual_peak = frames[np.argmax(actualh[mask])]
  knee_ref = np.rad2deg(c['reference_joint_pos'][:, knee])
  knee_actual = np.rad2deg(c['policy_joint_pos'][:, knee])
  knee_target = np.rad2deg(c['pd_target'][:, knee])
  # Use the original 50 Hz reference peak for every speed, interpolating actual
  # samples when retiming places that instant between control frames.
  with np.load(PROJECT_ROOT / meta['motion'], allow_pickle=False) as original:
    original_t = np.arange(len(original['joint_pos']))*meta['control_dt']
    original_foot = kin['body_names'].tolist().index('left_ankle_roll_link')
    original_mask = (original_t >= focus[0]-1e-8) & (original_t <= focus[1]+1e-8)
    original_frames = np.flatnonzero(original_mask)
    original_peak = original_frames[np.argmax(original['body_pos_w'][original_mask, original_foot, 2])]
    peak_t = float(original_t[original_peak])
    original_peak_height = float(original['body_pos_w'][original_peak, original_foot, 2])
    original_peak_knee = float(np.rad2deg(original['joint_pos'][original_peak, knee]))
  # An applied target is held over [control end - dt, control end), not after it.
  target_t = t - meta['control_dt']*meta['speed']
  ref_peak = frames[np.argmax(knee_ref[mask])]
  pre_frames = frames[frames <= ref_peak]
  trough = pre_frames[np.argmin(knee_ref[pre_frames])]
  threshold = (knee_ref[trough] + knee_ref[ref_peak])/2
  rising = (t >= t[trough]-1e-8) & (t <= t[ref_peak]+.1)
  tr = first_upcross(t[rising], knee_ref[rising], threshold)
  tt = first_upcross(target_t[rising], knee_target[rising], threshold, held=True)
  ta = first_upcross(t[rising], knee_actual[rising], threshold)
  kact = s['actuator_joint_names'].tolist().index('left_knee_joint')
  smask = (ts >= focus[0]-1e-8) & (ts <= focus[1]+1e-8)
  j = int(np.clip(np.searchsorted(ts, peak_t, side='right')-1,0,len(ts)-1))
  right = kin['body_names'].tolist().index('right_ankle_roll_link')
  left = kin['body_names'].tolist().index('left_ankle_roll_link')
  # The policy trajectory body list is a subset of the full FK body list.
  right_c = c['body_names'].tolist().index('right_ankle_roll_link')
  contacts = (s['foot_contact_count'] > 0) & (s['foot_force_n'] > 1)
  events = {}
  for side, i in (('left', 0), ('right', 1)):
    change = np.flatnonzero(contacts[1:, i] != contacts[:-1, i])+1
    events[side] = [{'reference_t': float(ts[v]), 'contact': bool(contacts[v, i])} for v in change]
  target_index = min(int(np.searchsorted(t, peak_t-1e-9)), len(t)-1)
  ref_com = np.array([np.interp(peak_t, kin['t'], kin['com_world'][:, dim]) for dim in range(3)])
  ref_ankle = np.array([np.interp(peak_t, kin['t'], kin['body_pos'][:, right, dim]) for dim in range(3)])
  actual_com = np.array([np.interp(peak_t, t, c['com_world'][:, dim]) for dim in range(3)])
  actual_ankle = np.array([np.interp(peak_t, t, c['policy_body_pos_world'][:, right_c, dim]) for dim in range(3)])
  return dict(completed_diagnostic_window=meta['completed_diagnostic_window'], termination=meta['termination'],
      peak_reference_t=peak_t, reference_ankle_peak_cm=original_peak_height*100,
      actual_ankle_at_reference_peak_cm=float(np.interp(peak_t, t, actualh)*100),
      actual_ankle_peak_cm=float(actualh[actual_peak]*100), actual_ankle_peak_reference_t=float(t[actual_peak]),
      ankle_peak_lag_reference_ms=float((t[actual_peak]-peak_t)*1000),
      ankle_peak_lag_wall_ms=float((t[actual_peak]-peak_t)/meta['speed']*1000),
      knee_at_ankle_reference_peak_deg=dict(reference=original_peak_knee,
          applied_target_preceding_interval=float(knee_target[target_index]), actual=float(np.interp(peak_t, t, knee_actual))),
      knee_rise=dict(threshold_deg=float(threshold), reference_cross_t=tr, target_cross_t=tt, actual_cross_t=ta,
          target_delay_reference_ms=(tt-tr)*1000 if tt is not None and tr is not None else None,
          actual_delay_reference_ms=(ta-tr)*1000 if ta is not None and tr is not None else None,
          definition='First upward crossing after reference trough; applied targets are zero-order held at interval starts; reference/actual crossings are linearly interpolated. Times in original reference seconds.'),
      knee_torque_max_nm=float(abs(s['actuator_force'][smask, kact]).max()),
      knee_torque_limit_nm=float(s['actuator_force_limits'][kact]),
      knee_95_percent_saturation_fraction=float(np.mean(abs(s['actuator_force'][smask, kact]) >= .95*s['actuator_force_limits'][kact])),
      pd_force_reconstruction_max_error_nm=float(abs(predicted_pd_force(s)-s['actuator_force']).max()),
      physics_samples=len(ts), physics_sample_period_s=float(np.median(np.diff(s['t']))),
      contact_at_ankle_reference_peak=contacts[j].tolist(), contact_sample_reference_t=float(ts[j]), contact_events=events,
      ref_com_minus_right_ankle_xy_cm=((ref_com-ref_ankle)[:2]*100).tolist(),
      actual_com_minus_right_ankle_xy_cm=((actual_com-actual_ankle)[:2]*100).tolist(),
      ref_com_height_cm=float(ref_com[2]*100), actual_com_height_cm=float(actual_com[2]*100),
      reference_knee_peak_speed_rad_s=float(abs(np.gradient(np.deg2rad(knee_ref), c['t']))[mask].max()),
      actual_knee_peak_speed_rad_s=float(abs(s['joint_vel'][smask, knee]).max()),
      reference_fk_ankle_max_difference_m=float(abs(np.interp(t[mask], kin['t'], kin['body_pos'][:, left, 2])-refh[mask]).max()))


def plot(runs, focus, path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(5, len(runs), figsize=(5*len(runs), 15), squeeze=False, sharey='row')
  for col, (prefix, meta, c, s, kin, report) in enumerate(runs):
    t, ts = c['reference_t'], s['reference_t']
    for row, part in enumerate(('knee', 'hip_pitch')):
      j = c['joint_names'].tolist().index(f'left_{part}_joint')
      ax = axes[row, col]
      ax.plot(t, np.rad2deg(c['reference_joint_pos'][:, j]), color='#57b7e9', label='Reference')
      ax.step(t-meta['control_dt']*meta['speed'], np.rad2deg(c['pd_target'][:, j]), where='post', color='#bd73be', label='Applied PD target', lw=1)
      ax.plot(t, np.rad2deg(c['policy_joint_pos'][:, j]), color='#258f62', label='Actual')
      ax.set_ylabel(f'Left {part} (deg)')
    ax = axes[2, col]
    j = c['body_names'].tolist().index('left_ankle_roll_link')
    ax.plot(t, c['reference_body_pos_world'][:, j, 2]*100, color='#57b7e9', label='Reference')
    ax.plot(t, c['policy_body_pos_world'][:, j, 2]*100, color='#258f62', label='Actual')
    ax.set_ylabel('Left ankle height (cm)')
    ax = axes[3, col]
    j = s['actuator_joint_names'].tolist().index('left_knee_joint')
    ax.plot(ts, s['actuator_force'][:, j], color='#cc8437', label=f'Knee torque ({1/float(s["physics_dt"]):g} Hz)')
    ax.axhline(s['actuator_force_limits'][j], color='#999999', ls='--', label='Torque limits')
    ax.axhline(-s['actuator_force_limits'][j], color='#999999', ls='--')
    ax.set_ylabel('Knee torque (Nm)')
    ax = axes[4, col]
    for i, side, color in ((0, 'Left', '#bd73be'), (1, 'Right', '#258f62')):
      ax.plot(ts, s['foot_force_n'][:, i], label=f'{side} strongest contact', color=color)
    ax.set_ylabel('Contact force norm (N)')
    ax.set_xlabel('Original reference time (s)')
    axes[0, col].set_title(f'{Path(prefix).name}\nReference speed {meta["speed"]:g}x', fontsize=10)
    for ax in axes[:, col]:
      ax.set_xlim(focus)
      ax.grid(alpha=.2)
      ax.legend(fontsize=8, loc='upper right')
  fig.suptitle('Left-leg control response — same physical timestep and PD gains; local diagnosis only')
  fig.tight_layout(rect=(0, 0, 1, .97))
  fig.savefig(path, dpi=130)
  plt.close(fig)
  fig, axes = plt.subplots(2, len(runs), figsize=(5*len(runs), 7), squeeze=False, sharey='row')
  for col, (_, meta, c, s, kin, _) in enumerate(runs):
    right = kin['body_names'].tolist().index('right_ankle_roll_link')
    actual = c['body_names'].tolist().index('right_ankle_roll_link')
    for dim in range(2):
      ax = axes[dim, col]
      ax.plot(kin['t'], (kin['com_world'][:, dim]-kin['body_pos'][:, right, dim])*100, color='#57b7e9', label='Reference')
      ax.plot(c['reference_t'], (c['com_world'][:, dim]-c['policy_body_pos_world'][:, actual, dim])*100, color='#258f62', label='Actual')
      ax.set_xlim(focus); ax.grid(alpha=.2); ax.legend()
      ax.set_ylabel(f'COM - right ankle {"XY"[dim]} (cm)')
      ax.set_xlabel('Original reference time (s)')
    axes[0, col].set_title(f'Reference speed {meta["speed"]:g}x')
  fig.suptitle('Whole-body COM relative to right ankle in world axes — not a stability margin')
  fig.tight_layout(rect=(0, 0, 1, .96))
  fig.savefig(path.with_name(path.stem+'_com.png'), dpi=130)
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--runs', nargs='+', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--focus', nargs=2, type=float, default=(5., 5.6))
  args = ap.parse_args()
  runs = []
  for prefix in args.runs:
    meta, c, s, kin = load(prefix)
    runs.append((prefix, meta, c, s, kin, analyze(meta, c, s, kin, args.focus)))
  if len({r[1]['motion_sha256'] for r in runs}) != 1:
    raise ValueError('comparison requires the same original reference')
  for field in ('task', 'seed', 'control_dt', 'physics_dt'):
    if len({r[1][field] for r in runs}) != 1:
      raise ValueError(f'comparison requires matching {field}')
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  report = dict(kind='control_diagnostic_comparison', quality_accepted=False, focus_reference_s=args.focus,
      runs={Path(p).name: dict(source=project_relative(p)+'.json', speed=m['speed'], **r) for p, m, _, _, _, r in runs},
      limitations=runs[0][1]['limitations'])
  Path(str(out)+'.json').write_text(json.dumps(report, indent=2)+'\n')
  plot(runs, args.focus, Path(str(out)+'.png'))
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
