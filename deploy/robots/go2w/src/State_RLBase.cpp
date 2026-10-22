#include <cmath>
#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "FSM/ThermalGuard.h"

namespace {
// Leg motor temperatures (see ThermalGuard); one per process like the state.
std::unique_ptr<ThermalGuard> thermal_guard;
}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // Overheated leg motors -> speed limit, stop, then lie down through Passive
    // before the firmware cuts them (see ThermalGuard).
    thermal_guard = std::make_unique<ThermalGuard>(param::config["ThermalGuard"], env.get());
    this->registered_checks.emplace_back(
        std::make_pair(
            []()->bool{ return thermal_guard && thermal_guard->critical(); },
            FSMStringMap.right.at("Passive")
        )
    );

    // Base tilt > 1.0 rad -> Passive. Checked here because the shared
    // isaaclab::mdp::bad_orientation is stubbed to always return false.
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{
                float cos_tilt = std::clamp(-env->robot->data.projected_gravity_b[2], -1.0f, 1.0f);
                return std::acos(cos_tilt) > 1.0f;
            },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    thermal_guard->update(lowstate);
    // Legs (first 12 policy joints) take position targets, wheels (last 4)
    // velocity targets; their kp is 0, so the motor applies kd * (dq_target - dq).
    // A braked wheel with brake_hold_kp also holds the angle it had when the
    // brake engaged (see JointVelocityAction), as in training.
    auto action = env->action_manager->processed_actions();
    const auto & ids = env->robot->data.joint_ids_map;
    for(int i(0); i < 12; i++) {
        lowcmd->msg_.motor_cmd()[ids[i]].q() = action[i];
    }
    const auto * wheels = env->action_manager->find_term<isaaclab::JointVelocityAction>();
    const bool hold = wheels && wheels->brake_hold_kp() > 0.0f && wheels->braked();
    for(int i(12); i < 16; i++) {
        auto & motor = lowcmd->msg_.motor_cmd()[ids[i]];
        motor.dq() = action[i];
        if(hold) {
            // Target within half a turn of the measured angle, so an angle that
            // wraps around cannot ask the motor for a full turn.
            const float q = lowstate->msg_.motor_state()[ids[i]].q();
            motor.q() = q + std::remainder(wheels->hold_pos(i - 12) - q, 2.0f * float(M_PI));
            motor.kp() = wheels->brake_hold_kp();
        } else {
            motor.kp() = 0.0f;
        }
    }
}
