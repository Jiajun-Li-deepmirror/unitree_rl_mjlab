#pragma once
#include "FSM/State_RLBase.h"
#include "MotionClip.h"
#include "PolicyWorker.h"
#include "FallGuard.h"
#include <fstream>

class State_Mimic : public FSMState {
public:
    State_Mimic(int state_mode, std::string state_string);
    void enter();
    void run();
    void exit() { worker.stop(); if (trace.is_open()) trace.close(); }
    using MotionLoader_ = MotionClip;
    static std::shared_ptr<MotionLoader_> motion;
private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_;
    std::unique_ptr<MotionWindow> window;
    PolicyWorker worker;
    FallGuard guard;
    std::ofstream trace;
};
REGISTER_FSM(State_Mimic)
