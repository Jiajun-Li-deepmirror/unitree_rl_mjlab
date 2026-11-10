# jobs/recipes/as2w.sh -- As2W's whole recipe for jobs/full_chain.sh (go2-depth 2026-11-10, the reproducible run; the user:
# "每个机型用仓库里的最终配方，从零跑完整条链，中间不做任何人工修正"). Taken from the logged pilots of this round's runs:
# blind a1g (the a1f + a1g average went out as as2w_blind_v3), depth a3m + backcons + gaitw4 (as2w_depth_v6_mg4).
# standpose (the payload stop stance, judged since 2026-11-10) is in every recipe.
# Blind: the T and M pilot (RECIPE_BLIND_TM), the P pilot (RECIPE_BLIND_P), the constraint multipliers at the start.
RECIPE_BLIND_TM=wheelclock,speedclock,critic,marchsettle,marchw40,standsettle,liftflat,sidew3,straightw3,marchsmooth,standsmooth,marchquiet,speedboth,grip,nohipload,lowent,arcclock,hipzero,marchreset,lowfric,stairboth,stairline,gaitw1,arcstep,sidew6,marchreset2,steptarget,diagaxis,riserprox,stairshare,drivepose,slowarc,drivelp,drcritic,brakeside,slowarc2,stairs50,stairhead,longexit,landings,wumore,payload,consease,sidecmd,sidetgt75,speedw4,straightcmd,trackmin,wheelcons,spinhold,spinflip,marchpose,marchclear,standpose
RECIPE_BLIND_P=wheelclock,speedclock,critic,marchsettle,marchw40,standsettle,liftflat,sidew3,straightw3,marchsmooth,standsmooth,marchquiet,speedboth,grip,nohipload,lowent,arcclock,hipzero,marchreset,lowfric,stairboth,stairline,gaitw1,arcstep,sidew6,marchreset2,steptarget,diagaxis,riserprox,stairshare,drivepose,slowarc,drivelp,drcritic,brakeside,slowarc2,stairs50,stairhead,longexit,landings,wumore,payload,consease,sidecmd,sidetgt75,speedw4,straightcmd,trackmin,wheelcons,spinhold,spinflip,marchpose,marchclear,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,liftm8,stairhead3,stairhead6,stairroll,tilt3,stairstab,stepclean,stairspeed4,paycorner,idlestep,stairstandw,standpose
RECIPE_BLIND_CONS=
# T's last iteration (the flat teacher, checked every 500 from 2000; 2026-11-10)
RECIPE_T_MAX=5000
# Depth (jobs/v2_chain.sh full): the pilot extras, the constraint start, the teacher's top-landing gate (% lifted, wheeled;
# empty = none), stage-1 distillation from scratch (iterations, learning rate).
RECIPE_DEPTH_EXTRA=,wheelstep,wheelstep4,stairspeed2,stairspeed3,noslide,noslide2x,wheelahead,topairup,longexit,liftm8,endwall,landings,wumore,payload,teachblur,backcons,gaitw4
RECIPE_DEPTH_CONS=
RECIPE_TOPFLAT=22
RECIPE_S1_ITERS=6000
RECIPE_S1_LR=5e-4
