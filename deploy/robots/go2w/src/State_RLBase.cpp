#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

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
    // Legs (first 12 policy joints) take position targets, wheels (last 4)
    // velocity targets; their kp is 0, so the motor applies kd * (dq_target - dq).
    auto action = env->action_manager->processed_actions();
    const auto & ids = env->robot->data.joint_ids_map;
    for(int i(0); i < 12; i++) {
        lowcmd->msg_.motor_cmd()[ids[i]].q() = action[i];
    }
    for(int i(12); i < 16; i++) {
        lowcmd->msg_.motor_cmd()[ids[i]].dq() = action[i];
    }
}
