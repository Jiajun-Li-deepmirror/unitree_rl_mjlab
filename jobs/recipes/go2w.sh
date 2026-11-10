# jobs/recipes/go2w.sh -- Go2W's whole recipe for jobs/full_chain.sh (go2-depth 2026-11-10, the reproducible run; the user:
# "每个机型用仓库里的最终配方，从零跑完整条链，中间不做任何人工修正"). Taken from the logged pilots of this round's runs:
# blind g27b (delivered go2w_blind_v3), depth w27d (go2w_depth_v6_w27d).
# standpose (the payload stop stance, judged since 2026-11-10) is in every recipe.
# Blind: the T and M pilot (RECIPE_BLIND_TM), the P pilot (RECIPE_BLIND_P), the constraint multipliers at the start.
# T / M (2026-11-11, rp4): the M pilot the delivered lineage trained with (g4 -> g14, M passed at 12600), i.e. without
# the stairs-quality keys (jobs/v2_robot.sh V2_STAIR_KEYS) and the later polish keys, which P adds. rp3 trained T / M
# on the whole P pilot (taken from g27b, a P-only run): at M +3000 / +3500 it fell at the first riser in every up band
# (0% up, wheel_riser_lift flat 4600 -> 8200) where g4 already climbed at +3000 (only the stairs speeds missed).
RECIPE_BLIND_TM=wheelclock,speedclock,critic,marchsettle,marchw40,standsettle,liftflat,sidew3,straightw3,marchsmooth,standsmooth,marchquiet,speedboth,grip,nohipload,lowent,arcclock,hipzero,marchreset,lowfric,stairboth,stairline,gaitw1,arcstep,sidew6,marchreset2,steptarget,diagaxis,riserprox,stairshare,drivepose,slowarc,drivelp,stairs50,stairhead,wheelcons,longexit,landings,wumore,payload,sidetgt75,sidecmd,speedw4,straightcmd,trackmin,consease,standpose
RECIPE_BLIND_P=wheelclock,speedclock,critic,marchsettle,marchw40,standsettle,liftflat,sidew3,straightw3,marchsmooth,standsmooth,marchquiet,speedboth,grip,nohipload,lowent,arcclock,hipzero,marchreset,lowfric,stairboth,stairline,gaitw1,arcstep,sidew6,marchreset2,steptarget,diagaxis,riserprox,stairshare,drivepose,slowarc,drivelp,stairs50,stairhead,wheelcons,longexit,landings,wumore,payload,sidetgt75,sidecmd,speedw4,straightcmd,trackmin,consease,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,liftm8,stairspeed4,marchpose,marchclear,spinhold,stepclean,stairroll,tilt3,stairstab,spinflip,stairhead3,stairhead6,paycorner,standpose
RECIPE_BLIND_CONS=
# T's last iteration (the flat teacher, checked every 500 from 2000; 2026-11-10)
RECIPE_T_MAX=5000
# M's and P's length (2026-11-11; default 5000 / 2000; each stops at its first pass): the delivered lineage needed
# M 2000 -> 12600 and P 12600 -> 19800
RECIPE_M_ITERS=10000
RECIPE_P_ITERS=6000
# Depth (jobs/v2_chain.sh full): the pilot extras, the constraint start, the teacher's top-landing gate (% lifted, wheeled;
# empty = none), stage-1 distillation from scratch (iterations, learning rate).
RECIPE_DEPTH_EXTRA=,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,longexit,liftm8,landings,wumore,payload,teachblur,stairspeed5
RECIPE_DEPTH_CONS=
RECIPE_TOPFLAT=22
RECIPE_S1_ITERS=6000
RECIPE_S1_LR=5e-4
