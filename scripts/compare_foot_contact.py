"""Compare two complete, synchronized contact-audit rollouts and their peak lifts."""
import argparse
import json
from pathlib import Path

import numpy as np
from project_paths import project_relative
from tracking_diagnostics import summarize_contact


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path, required=True, help='trajectory NPZ with adjacent evaluation JSON')
  parser.add_argument('--candidate', type=Path, required=True)
  parser.add_argument('--out', type=Path, required=True, help='output prefix')
  args = parser.parse_args()
  trajectories, reports = [], []
  for path in (args.baseline, args.candidate):
    with np.load(path, allow_pickle=False) as archive:
      trajectories.append(dict(archive))
    reports.append(json.loads(path.with_suffix('.json').read_text()))
  b, c = trajectories
  for report in reports:
    if report['num_envs'] != 1 or report['completed_envs'] != 1:
      parser.error('contact comparison requires two complete single-environment rollouts')
  for key in ('motion_sha256', 'randomize', 'seed', 'task'):
    if reports[0][key] != reports[1][key]:
      parser.error(f'unmatched evaluation field: {key}')
  for key in ('t', 'reference_joint_pos', 'reference_body_pos_world', 'joint_names', 'body_names'):
    if not np.array_equal(b[key], c[key]):
      parser.error(f'unsynchronized trajectory field: {key}')
  result = {'baseline': project_relative(args.baseline), 'candidate': project_relative(args.candidate),
            'motion_sha256': reports[0]['motion_sha256'],
            'baseline_checkpoint_sha256': reports[0]['checkpoint_sha256'],
            'candidate_checkpoint_sha256': reports[1]['checkpoint_sha256'],
            'baseline_contact': summarize_contact(b), 'candidate_contact': summarize_contact(c),
            'peak_lifts': {}, 'definitions': {
              'height': 'Ankle body-origin height, not sole clearance.',
              'peak_time_offset': 'Local policy height peak within +/-0.6 s of the reference peak; not a fitted trajectory time shift.',
              'sampling': '50 Hz policy steps, strongest loaded contact per foot; not every contact point or physics substep.'}}
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(4, 2, figsize=(14, 12))
  t = b['t']
  for i, side in enumerate(('left', 'right')):
    body = b['body_names'].tolist().index(f'{side}_ankle_roll_link')
    height = b['reference_body_pos_world'][:, body, 2]
    peak = int(height.argmax())
    window = (t >= t[peak] - .6) & (t <= t[peak] + .6)
    lift = {'reference_peak_t': float(t[peak]), 'reference_peak_height_m': float(height[peak])}
    for data, name, color in ((b, 'baseline', '#777777'), (c, 'candidate', '#e27933')):
      actual = data['policy_body_pos_world'][:, body, 2]
      local_peak = np.flatnonzero(window)[actual[window].argmax()]
      knee = data['joint_names'].tolist().index(f'{side}_knee_joint')
      lift[name] = {'height_at_reference_peak_m': float(actual[peak]),
          'local_peak_height_m': float(actual[local_peak]),
          'local_peak_t': float(t[local_peak]), 'local_peak_time_offset_s': float(t[local_peak]-t[peak]),
          'contact_at_reference_peak': bool(data['foot_contact'][peak, i]),
          'force_at_reference_peak_n': float(data['foot_contact_force_n'][peak, i]),
          'knee_reference_deg': float(np.rad2deg(data['reference_joint_pos'][peak, knee])),
          'knee_policy_deg': float(np.rad2deg(data['policy_joint_pos'][peak, knee]))}
      axes[0,i].plot(t, actual*100, color=color, label=name)
      axes[1,i].plot(t[window], actual[window]*100, color=color, label=name)
      axes[2,i].step(t[window], data['foot_contact'][window,i], where='post', color=color, label=name)
      speed = np.linalg.norm(data['foot_contact_point_velocity_world'][:,i,:2], axis=-1)
      axes[3,i].plot(t[window], np.where(data['foot_contact'][:,i], speed, np.nan)[window], color=color, label=name)
    for row, mask in ((0, slice(None)), (1, window)):
      axes[row,i].plot(t[mask], height[mask]*100, color='#69b9ee', label='reference')
      axes[row,i].set_ylabel('ankle height (cm)')
    axes[0,i].set_title(f'{side}: complete motion')
    axes[1,i].set_title(f'{side}: reference maximum lift at {t[peak]:.2f} s')
    axes[2,i].set_ylabel('measured contact (>1 N)')
    axes[3,i].set_ylabel('contact point speed (m/s)')
    result['peak_lifts'][side] = lift
  for ax in axes.flat:
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=8)
    ax.set_xlabel('time (s)')
  fig.suptitle('Measured ground contact and lift tracking — synchronized world coordinates')
  fig.tight_layout()
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out.with_suffix('.png'), dpi=130)
  result['plot'] = project_relative(args.out.with_suffix('.png'))
  args.out.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps(result['peak_lifts'], indent=2))


if __name__ == '__main__':
  main()
