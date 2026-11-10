# jobs/recipes/as2.sh -- As2's whole recipe for jobs/full_chain.sh (go2-depth 2026-11-10, the reproducible run; the user:
# "每个机型用仓库里的最终配方，从零跑完整条链，中间不做任何人工修正"). Taken from the logged pilots of this round's runs:
# blind s1q (this round; no delivery yet), depth: the legged default.
# standpose (the payload stop stance, judged since 2026-11-10) is in every recipe.
# Blind: the T and M pilot (RECIPE_BLIND_TM), the P pilot (RECIPE_BLIND_P), the constraint multipliers at the start.
RECIPE_BLIND_TM=speedclock,critic,marchsettle,marchw40,standsettle,sidew3,straightw3,marchsmooth,standsmooth,marchquiet,speedboth,grip,nohipload,lowent,arcclock,hipzero,marchreset,lowfric,stairboth,stairline,stairs50,stairpay16,gaitw1,feetedge,payload,landings,wumore,spinflip,spinhold,spintight,spinup,standlong,stopfast,stopside,yawslew,flatorient3,basetouch,standpose
RECIPE_BLIND_P=speedclock,critic,marchsettle,marchw40,standsettle,sidew3,straightw3,marchsmooth,standsmooth,marchquiet,speedboth,grip,nohipload,lowent,arcclock,hipzero,marchreset,lowfric,stairboth,stairline,stairs50,stairpay16,gaitw1,feetedge,payload,landings,wumore,spinflip,spinhold,spintight,spinup,standlong,stopfast,stopside,yawslew,flatorient3,basetouch,standpose,stumble4x,feetmid,footslide,stairstab,edgecons,feetonce,feetonce2,stairline2,stairroll,stairshare,stairstand4,support2,tilt3
RECIPE_BLIND_CONS=
# T's last iteration (the flat teacher, checked every 500 from 2000; 2026-11-10)
RECIPE_T_MAX=5000
# Depth (jobs/v2_chain.sh full): the pilot extras, the constraint start, the teacher's top-landing gate (% lifted, wheeled;
# empty = none), stage-1 distillation from scratch (iterations, learning rate).
RECIPE_DEPTH_EXTRA=,longexit,endwall,teachblur
RECIPE_DEPTH_CONS=
RECIPE_TOPFLAT=
RECIPE_S1_ITERS=6000
RECIPE_S1_LR=5e-4
