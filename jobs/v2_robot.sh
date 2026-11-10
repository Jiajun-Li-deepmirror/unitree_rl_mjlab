# v2_robot.sh R (sourced; go2-depth 2026-11-05, the new pipeline): per-robot inputs of the v2 recipe. The recipe itself
# (V2_WHEELED_EXTRA below) is the same for every wheeled robot; only what differs by robot lives here.
#   V2_PKG_VAR / V2_PKG  the blind delivery package the depth teacher descends from (fe_robot.sh's <R>_PKG override)
#   V2_TEACHER_FROM      raw teacher checkpoint (with optimizer) the v2 teacher fine-tunes from
#   V2_STUDENT_FROM      student the stage-1 distillation warm-starts from (the best accepted / deployed so far)
#   V2_CONS              constraint multipliers to start from (FE_CONS_INIT)
#   V2_ROBOT_EXTRA       robot-only pilot keys on top of the recipe
# Payload limits and the wall stop distance come from fe_robot.sh (BLIND_PAYLOAD, BLIND_WALL_STOP).
D=/root/gpufree-data/go2-depth/depth_fe
# The wheeled v2 recipe (the user's stairs requirements, 11-05): per-wheel flat judgement looking ahead (wheelstep,
# wheelahead), near-stairs lifts 4x (wheelstep4), slide-back and riser contact charged at 2x (noslide, noslide2x),
# lift margin 8 cm (liftm8), the climb's top landing in the air charged (topairup), stairs speed deficit 3x
# (stairspeed2, stairspeed3), 1.4 m exits closed by walls with the forward command slowing in front of any wall
# (longexit, endwall), landings in half the up flights and more of them (landings, wumore), payload up to the
# documented limit with a wide CoM (payload), and the teacher's scan as the camera sees it (teachblur).
V2_WHEELED_EXTRA=,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,longexit,liftm8,endwall,landings,wumore,payload,teachblur
# The blind stage's recipes (blind_chain.sh): what of the above a blind policy can use (no endwall: it cannot see a
# wall; no teachblur: no scan), and the legged one for the user's legged stairs requirements (11-05): riser kicks
# (stumble4x), feet in the tread's middle (feetmid), no foot sliding back down a step (footslide), a steady body on
# stairs (stairstab, both kinds), payload, landings in the up flights.
V2_WHEELED_BLIND_EXTRA=,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,longexit,liftm8,landings,wumore,payload
V2_LEGGED_BLIND_EXTRA=,payload,stumble4x,feetmid,footslide,stairstab,landings,wumore
# The legged depth recipe (11-06): the legged blind delivery's pilot already carries the blind keys above plus its polish
# fixes (B2: sidew6l, stairline2); the depth teacher adds what needs eyes: walls closing the 1.4 m exits with the
# command slowing in front of them (longexit, endwall) and the scan as the camera sees it (teachblur).
V2_LEGGED_EXTRA=,longexit,endwall,teachblur
# Stairs-quality keys (11-05 22:30): the blind chain adds them only in its polish phase. From zero with them all along the
# Go2W blind (T avg3500, M avg6500) never climbed: every up-stairs robot ended at the first riser on base / hip contact
# (illegal_contact, 5-18 cm alike) while go2w_polish_11300 climbed 0/192 ended in the same env; riser contact at 6/s,
# slide-back 4 per slide and 4x lifts made the stair tiles cheaper to end than to climb. They were built to fine-tune
# policies that climb (the depth teachers), so: T / M learn to climb, P adds them. No observation changes.
V2_STAIR_KEYS="wheelstep wheelstep4 stairspeed2 stairspeed3 stairspeed4 noslide noslide2x wheelahead topairup liftm8 stumble4x feetmid footslide stairstab"
# legged packages: logs/pkg/<name> where copied (t1), else go2-blind's export (t2)
lpkg() { if [ -d $D/logs/pkg/$1 ]; then echo $D/logs/pkg/$1; else echo /root/gpufree-data/go2-blind/export/legged_feetedge/$1; fi; }
case $1 in *W) V2_KIND=wheeled ;; *) V2_KIND=legged ;; esac
case $1 in
  # Go2W (11-06): the new pipeline's blind go2w_bl2_g14_13600 (its pilot carries the flat fixes, consease and the
  # polish stairs keys); no constraint start at 2x and no sidetgt75 (pinned guards overshot the sideways speed).
  # Go2W (11-07): the blind of the g17-g20b polish rounds (march lift, stairs posture, spin hold, leg hold; tinynav
  # slide-back 4.4%); was go2w_bl2_g14_13600
  # Go2W (11-09): go2w_bl2_g27b_19800 (g26 19300 + paycorner, +500; mjlab s0-2 PASS with the loaded stand judged
  # after 3 s (user 11-08); tinynav slide-back 5.0%, heading 2.8 deg, 12 kg slide-back 8.9%, no falls); was g20b_18300
  Go2W) V2_PKG_VAR=GO2W_PKG V2_PKG=$D/logs/pkg/go2w_bl2_g27b_19800
        V2_TEACHER_FROM=$D/logs/rsl_rl/go2w_depth_teacher/2026-11-05_04-53-21_tf_v5/model_26200.pt
        V2_STUDENT_FROM=$D/logs/rsl_rl/go2w_depth_student/fe_edge_tf5s_avg1499/model_1499.pt
        V2_CONS= V2_ROBOT_EXTRA=
        export GO2W_N=19800 ;;
  # As2W (11-09): as2w_bl2_mixfg_11000 (the blind chain's a1f avg10500 + a1g avg11000 average; mjlab s0-2 PASS, tinynav
  # on the grip solver slide-back 2.8%, 13 kg 0.0%, no falls); was go2-home's as2w_mixed_8000
  As2W) V2_PKG_VAR=AS2W_PKG V2_PKG=$D/logs/pkg/as2w_bl2_mixfg_11000
        V2_TEACHER_FROM=$D/logs/rsl_rl/as2w_depth_teacher/2026-11-05_11-36-08_tf_v8/model_17800.pt
        V2_STUDENT_FROM=$D/logs/rsl_rl/as2w_depth_student/fe_edge_tf6p_avg1000/model_1000.pt
        V2_CONS= V2_ROBOT_EXTRA=
        export AS2W_N=11000 AS2W_PH=Polish AS2W_PSTART= ;;
  B2W)  V2_PKG_VAR=B2W_PKG V2_PKG=$D/logs/pkg/b2w_polish_9000_sidecmd_desc
        V2_TEACHER_FROM=$D/logs/rsl_rl/b2w_depth_teacher/2026-11-04_23-58-35_tf_v3/model_15900.pt
        V2_STUDENT_FROM=$D/logs/rsl_rl/b2w_depth_student/fe_edge_tf3e_avg1000/model_1000.pt
        V2_CONS=back_drift:1.7,descent:2.3 V2_ROBOT_EXTRA=,bdhold ;;
  # Legged (2026-11-05, the user: legged too, from zero, on t2 once g1-loc hands it back): go2-home's feetedge
  # deliveries (Go2's on t1's export only, As2's on t2's, B2's on both).
  Go2)  V2_PKG_VAR=GO2_PKG V2_PKG=$(lpkg go2_fe_polish_10000) V2_CONS=clock_side:3,side_drift:3,feet_edge:2 V2_ROBOT_EXTRA=
        V2_TEACHER_FROM= V2_STUDENT_FROM= ;;
  As2)  V2_PKG_VAR=AS2_PKG V2_PKG=$(lpkg as2_fe_polish_12000) V2_CONS= V2_ROBOT_EXTRA= V2_TEACHER_FROM= V2_STUDENT_FROM= ;;
  # B2 (11-06): the new pipeline's blind delivery (blind_chain l4, avg9000, a polish package; its pilot is also the
  # blind chain's base from here on); before it go2-home's b2_fe_mixed_5000 (B2_N=5000 B2_PH=Mixed).
  # Student warm start: fe_edge_v4rl avg1500 (11-02, the RL-tuned feetedge student; copied from t1 11-06), the only B2
  # student checkpoint left: fe_chain's default (logs/occ/b2_occ_v1_model_1999.pt) and t2's students were cleaned up.
  # B2 (11-07): the blind of the l5-l8 polish rounds (b2_bl2_l8_11800: kicks 2.4, map3_up 3.0, mid 77.6%, spin_alt pass;
  # was b2_bl2_l4_9000)
  # B2 (11-07 17:50): b2_bl2_l10_12300 (l8 + stairstand4: stopped mid-descent on four feet 0.95-1.00, l8 0.29-0.46;
  # mjlab s0-2 PASS incl. the stop-feet and loaded-stand items; tinynav 8 runs kicks 3.3, mid 77.8%, slips 0.2)
  B2)   V2_PKG_VAR=B2_PKG V2_PKG=$D/logs/pkg/b2_bl2_l10_12300 V2_CONS= V2_ROBOT_EXTRA= V2_TEACHER_FROM=
        V2_STUDENT_FROM=$D/logs/rsl_rl/b2_depth_student/fe_edge_v4rl_avg1500/model_1500.pt
        export B2_N=12300 B2_PH=Polish; unset B2_PSTART ;;
  *) echo "v2_robot.sh: no v2 inputs for $1" >&2; return 2 ;;
esac
