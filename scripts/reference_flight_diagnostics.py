"""Necessary free-flight COM consistency check on a flat, stationary floor."""
import json
from pathlib import Path

import numpy as np


def flight_intervals(t, com, gap, gravity, window, speed=1., threshold=.01):
  mask = (gap > threshold) & (t*speed >= window[0]) & (t*speed <= window[1])
  ids = np.flatnonzero(mask)
  segments = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
  result = []
  for frames in segments:
    if len(frames) < 5:
      continue
    x = t[frames]-t[frames[0]]
    z = com[frames, 2]
    # Fit over a sustained interval to avoid interpreting one noisy second difference.
    fit = np.polyfit(x, z, 2)
    ballistic = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)),
                               z-.5*gravity[2]*x*x, rcond=None)[0]
    residual = z-(ballistic[0]+ballistic[1]*x+.5*gravity[2]*x*x)
    result.append(dict(start_reference_t=float(t[frames[0]]*speed),
        end_reference_t=float(t[frames[-1]]*speed), samples=len(frames),
        duration_wall_s=float(x[-1]), minimum_all_collision_clearance_cm=float(gap[frames].min()*100),
        fitted_com_vertical_acceleration_m_s2=float(2*fit[0]),
        gravity_vertical_m_s2=float(gravity[2]),
        equivalent_upward_external_force_in_bodyweights=float((2*fit[0]-gravity[2])/-gravity[2]),
        best_ballistic_fit_max_residual_cm=float(abs(residual).max()*100),
        quadratic_fit_max_residual_cm=float(abs(z-np.polyval(fit, x)).max()*100)))
  return result


def save_flight_diagnostics(kin, dt, speed, window, out):
  t = np.arange(len(kin['com_world']))*dt
  threshold = max(.01, float(kin['max_contact_margin'])+.005)
  intervals = flight_intervals(t, kin['com_world'], kin['robot_min_collision_height'],
                              kin['gravity'], window, speed, threshold)
  report = dict(kind='reference_free_flight_consistency', diagnostic_only=True,
      speed=speed, window_reference_s=list(window), clearance_threshold_m=threshold,
      robot_mass_kg=float(kin['robot_mass']), intervals=intervals,
      definition='All robot collision spheres/capsules exceed ground/contact margin. During true free flight, whole-body COM acceleration must equal gravity. Fit vertical COM acceleration over sustained >=5-frame intervals.',
      limitations=['This is a necessary free-flight check, not a proof of feasibility during stance.',
                  'Retargeted reference is sampled/filtered; fitted acceleration is an estimate.',
                  'Air resistance and external supports are absent in this training environment.'])
  Path(str(out)+'_reference_dynamics.json').write_text(json.dumps(report, indent=2)+'\n')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
  for i, label in enumerate(('Left foot', 'Right foot')):
    axes[0].plot(t*speed, kin['foot_min_collision_height'][:, i]*100, label=label)
  axes[0].axhline(threshold*100, color='grey', ls='--', label='Flight test threshold')
  axes[0].set_ylabel('Lowest collision point (cm)')
  axes[1].plot(t*speed, kin['com_world'][:, 2]*100, label='Reference whole-body COM')
  axes[1].set_ylabel('COM height (cm)')
  ballistic_heights = []
  for row in intervals:
    mask = (t*speed >= row['start_reference_t']-1e-8) & (t*speed <= row['end_reference_t']+1e-8)
    x = t[mask]-t[mask][0]; z = kin['com_world'][mask, 2]; g = kin['gravity'][2]
    fit = np.linalg.lstsq(np.column_stack((np.ones(len(x)),x)),z-.5*g*x*x,rcond=None)[0]
    height=(fit[0]+fit[1]*x+.5*g*x*x)*100
    ballistic_heights.extend(height.tolist())
    axes[1].plot(t[mask]*speed,height,'--',color='#cc8437')
    for ax in axes:
      ax.axvspan(row['start_reference_t'], row['end_reference_t'], color='#e0a645', alpha=.16)
  axes[1].plot([],[], '--', color='#cc8437', label='Best free-flight parabola')
  acceleration = np.gradient(np.gradient(kin['com_world'][:, 2],dt),dt)
  axes[2].plot(t*speed, acceleration, label='Reference COM acceleration (finite difference)')
  axes[2].axhline(kin['gravity'][2], color='#cc8437', ls='--', label='Gravity')
  axes[2].set_ylabel('Vertical acceleration (m/s²)')
  axes[2].set_xlabel('Original reference time (s)')
  for ax in axes:
    ax.set_xlim(window); ax.legend(loc='upper right'); ax.grid(alpha=.2)
  # Scale only to the inspected window, not remote extremes elsewhere in the motion.
  mask=(t*speed>=window[0]) & (t*speed<=window[1])
  heights=np.r_[kin['com_world'][mask,2]*100,ballistic_heights]
  axes[1].set_ylim(heights.min()-1,heights.max()+1)
  axes[2].set_ylim(min(acceleration[mask].min(),kin['gravity'][2])-2,max(acceleration[mask].max(),0)+2)
  fig.suptitle('Reference contact / COM consistency — shaded regions have no robot-ground contact geometry')
  fig.tight_layout(rect=(0,0,1,.97))
  fig.savefig(Path(str(out)+'_reference_dynamics.png'),dpi=140)
  plt.close(fig)
  return report
