#pragma once
#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "PolicyWorker.h"

class State_RLBase : public FSMState {
public:
    State_RLBase(int state_mode, std::string state_string);
    void enter() {
        env->reset();
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i) {
            auto& motor = lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]];
            motor.kp() = env->robot->data.joint_stiffness[i];
            motor.kd() = env->robot->data.joint_damping[i];
            motor.q() = env->robot->data.joint_pos[i];
            motor.dq() = motor.tau() = 0;
        }
        worker.start(env.get());
    }
    void run();
    void exit() { worker.stop(); }
private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    PolicyWorker worker;
};
REGISTER_FSM(State_RLBase)
