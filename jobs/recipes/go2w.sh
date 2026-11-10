# jobs/recipes/go2w.sh -- Go2W's whole recipe for jobs/full_chain.sh (go2-depth 2026-11-10, the reproducible run; the user:
# "每个机型用仓库里的最终配方，从零跑完整条链，中间不做任何人工修正"). Taken from the logged pilots of this round's runs:
# blind g27b (delivered go2w_blind_v3), depth w27d (go2w_depth_v6_w27d).
# standpose (the payload stop stance, judged since 2026-11-10) is in every recipe.
# Blind: the T and M pilot (RECIPE_BLIND_TM), the P pilot (RECIPE_BLIND_P), the constraint multipliers at the start.
RECIPE_BLIND_TM=wheelclock,speedclock,critic,marchsettle,marchw40,standsettle,liftflat,sidew3,straightw3,marchsmooth,standsmooth,marchquiet,speedboth,grip,nohipload,lowent,arcclock,hipzero,marchreset,lowfric,stairboth,stairline,gaitw1,arcstep,sidew6,marchreset2,steptarget,diagaxis,riserprox,stairshare,drivepose,slowarc,drivelp,stairs50,stairhead,wheelcons,longexit,landings,wumore,payload,sidetgt75,sidecmd,speedw4,straightcmd,trackmin,consease,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,liftm8,stairspeed4,marchpose,marchclear,spinhold,stepclean,stairroll,tilt3,stairstab,spinflip,standpose
RECIPE_BLIND_P=wheelclock,speedclock,critic,marchsettle,marchw40,standsettle,liftflat,sidew3,straightw3,marchsmooth,standsmooth,marchquiet,speedboth,grip,nohipload,lowent,arcclock,hipzero,marchreset,lowfric,stairboth,stairline,gaitw1,arcstep,sidew6,marchreset2,steptarget,diagaxis,riserprox,stairshare,drivepose,slowarc,drivelp,stairs50,stairhead,wheelcons,longexit,landings,wumore,payload,sidetgt75,sidecmd,speedw4,straightcmd,trackmin,consease,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,liftm8,stairspeed4,marchpose,marchclear,spinhold,stepclean,stairroll,tilt3,stairstab,spinflip,stairhead3,stairhead6,paycorner,standpose
RECIPE_BLIND_CONS=
# Depth (jobs/v2_chain.sh full): the pilot extras, the constraint start, the teacher's top-landing gate (% lifted, wheeled;
# empty = none), stage-1 distillation from scratch (iterations, learning rate).
RECIPE_DEPTH_EXTRA=,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,longexit,liftm8,landings,wumore,payload,teachblur,stairspeed5
RECIPE_DEPTH_CONS=
RECIPE_TOPFLAT=22
RECIPE_S1_ITERS=6000
RECIPE_S1_LR=5e-4
