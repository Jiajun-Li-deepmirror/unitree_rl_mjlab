# jobs/recipes/b2w.sh -- B2W's whole recipe for jobs/full_chain.sh (go2-depth 2026-11-10, the reproducible run; the user:
# "每个机型用仓库里的最终配方，从零跑完整条链，中间不做任何人工修正"). Taken from the logged pilots of this round's runs:
# blind: go2-home's b2w_polish_9000 pilot + the wheeled v2 keys + bdhold (never run through this pipeline), depth: the wheeled default.
# standpose (the payload stop stance, judged since 2026-11-10) is in every recipe.
# Blind: the T and M pilot (RECIPE_BLIND_TM), the P pilot (RECIPE_BLIND_P), the constraint multipliers at the start.
RECIPE_BLIND_TM=wheelclock,speedclock,critic,marchsettle,marchw40,standsettle,liftflat,sidew3,straightw3,marchsmooth,standsmooth,marchquiet,speedboth,grip,nohipload,lowent,arcclock,hipzero,marchreset,lowfric,stairboth,stairline,gaitw1,arcstep,sidew6,marchreset2,steptarget,diagaxis,marchgait3,standlongw,standw4w,riserprox,stairshare,drivepose,slowarc,drivelp,drcritic,brakeside,slowarc2,legkp15,standtgt,restlock,leghold20,stairs50,stairhead,stairs35,b2wcons,sidecmd,b2wdesc,longexit,landings,wumore,payload,bdhold,standpose
RECIPE_BLIND_P=wheelclock,speedclock,critic,marchsettle,marchw40,standsettle,liftflat,sidew3,straightw3,marchsmooth,standsmooth,marchquiet,speedboth,grip,nohipload,lowent,arcclock,hipzero,marchreset,lowfric,stairboth,stairline,gaitw1,arcstep,sidew6,marchreset2,steptarget,diagaxis,marchgait3,standlongw,standw4w,riserprox,stairshare,drivepose,slowarc,drivelp,drcritic,brakeside,slowarc2,legkp15,standtgt,restlock,leghold20,stairs50,stairhead,stairs35,b2wcons,sidecmd,b2wdesc,longexit,landings,wumore,payload,bdhold,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,liftm8,standpose
RECIPE_BLIND_CONS=back_drift:1.7,descent:2.3
# Depth (jobs/v2_chain.sh full): the pilot extras, the constraint start, the teacher's top-landing gate (% lifted, wheeled;
# empty = none), stage-1 distillation from scratch (iterations, learning rate).
RECIPE_DEPTH_EXTRA=,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,longexit,liftm8,endwall,landings,wumore,payload,teachblur,bdhold
RECIPE_DEPTH_CONS=back_drift:1.7,descent:2.3
RECIPE_TOPFLAT=
RECIPE_S1_ITERS=6000
RECIPE_S1_LR=5e-4
