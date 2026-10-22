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
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}