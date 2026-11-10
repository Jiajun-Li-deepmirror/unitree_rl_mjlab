#include "State_Mimic.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

static Eigen::Quaternionf init_quat;
std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;


Eigen::Quaternionf robot_quat_w(isaaclab::ManagerBasedRLEnv* env)
{
    using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    G1Type* robot = dynamic_cast<G1Type*>(env->robot.get());

    auto root_quat = env->robot->data.root_quat_w;
    const auto& joints = env->robot->data.joint_pos;

    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(joints[12], Eigen::Vector3f::UnitZ()) \
        * Eigen::AngleAxisf(joints[13], Eigen::Vector3f::UnitX()) \
        * Eigen::AngleAxisf(joints[14], Eigen::Vector3f::UnitY());

//    return root_quat;
    return torso_quat;
}

Eigen::Quaternionf motion_anchor_quat_w(std::shared_ptr<State_Mimic::MotionLoader_> loader)
{
    const auto root_quat = loader->root_quaternion();
    const auto joint_pos = loader->joint_pos();
    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(joint_pos[12], Eigen::Vector3f::UnitZ()) \
        * Eigen::AngleAxisf(joint_pos[13], Eigen::Vector3f::UnitX()) \
        * Eigen::AngleAxisf(joint_pos[14], Eigen::Vector3f::UnitY());

//    return root_quat;
    return torso_quat;
}


namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_command)
{
    auto loader = State_Mimic::motion;
    std::vector<float> data;

    auto motion_joint_pos = loader->joint_pos();
    auto motion_joint_vel = loader->joint_vel();

    data.insert(data.end(),
                motion_joint_pos.data(),
                motion_joint_pos.data() + motion_joint_pos.size());
    data.insert(data.end(),
                motion_joint_vel.data(),
                motion_joint_vel.data() + motion_joint_vel.size());
    return data;
}

REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    auto loader = State_Mimic::motion;
    std::vector<float> out;

    auto real_quat_w = robot_quat_w(env);
    auto ref_quat_w  = motion_anchor_quat_w(loader);

    auto rot_ = (init_quat * ref_quat_w).conjugate() * real_quat_w;
    // Materialize the transpose: `auto` would retain an Eigen expression
    // referring to a destroyed temporary matrix and corrupt the observation.
    const Eigen::Matrix3f rot = rot_.toRotationMatrix().transpose();

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}

}
}


State_Mimic::State_Mimic(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if(!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }

    // Motion
    motion_ = std::make_shared<MotionLoader_>(motion_file.string());
    spdlog::info("Loaded motion file '{}' with duration {:.2f}s", motion_file.stem().string(), motion_->duration);
    motion = motion_;
    window = std::make_unique<MotionWindow>(
        cfg["time_start"].as<float>(0.0f), cfg["time_end"].as<float>(motion_->duration), motion_->duration);
    guard = FallGuard(cfg["fall_guard"]);
    std::string end_state = "Velocity";
    if (cfg["end_state"]) {
        end_state = cfg["end_state"].as<std::string>();
    }

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
    if (std::abs(env->step_dt - MotionClip::dt) > 1e-6f)
        throw std::runtime_error("Mimic policy and reference must both run at 50 Hz");

    const auto & joy = FSMState::lowstate->joystick;
    this->registered_check_labels[this->registered_checks.size()] = "motion_complete";
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return window->finished(worker.steps.load(), env->step_dt); }, // time out
            FSMStringMap.right.at(end_state)
        )
    );
    this->registered_check_labels[this->registered_checks.size()] = "policy_fault";
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return worker.unhealthy(); }, // bad orientation
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_Mimic::enter()
{
    guard.reset();
    motion = motion_;
    motion->update_frame(window->first_frame);
    env->reset();
    // Align anchor yaw to anchor yaw (the reference pelvis may have waist yaw).
    init_quat = isaaclab::yawQuaternion(robot_quat_w(env.get())) *
                isaaclab::yawQuaternion(motion_anchor_quat_w(motion)).conjugate();
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i) {
        auto& motor = lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]];
        motor.kp() = env->robot->data.joint_stiffness[i];
        motor.kd() = env->robot->data.joint_damping[i];
        motor.q() = env->robot->data.joint_pos[i];
        motor.dq() = motor.tau() = 0;
    }
    auto trace_dir = param::config["trace_dir"];
    if (trace_dir) {
        auto directory = param::resolve_path(trace_dir.as<std::string>());
        std::filesystem::create_directories(directory);
        trace.open(directory / (getStateString() + ".csv"));
        trace << "step,frame,reference_time,inference_s";
        trace << ",root_w,root_x,root_y,root_z,gyro_x,gyro_y,gyro_z,init_w,init_x,init_y,init_z";
        for (int i = 0; i < 29; ++i) trace << ",q" << i;
        for (int i = 0; i < 29; ++i) trace << ",dq" << i;
        for (int i = 0; i < 29; ++i) trace << ",target" << i;
        for (int i = 0; i < 29; ++i) trace << ",action" << i;
        for (int i = 0; i < 154; ++i) trace << ",obs" << i;
        trace << "\n";
        trace << std::setprecision(9);
    }
    spdlog::info("Motion window: [{:.3f}, {:.3f}), {:.3f} seconds",
                 window->start, window->end, window->end - window->start);
    worker.start(env.get(), [this] {
        motion->update_frame(window->first_frame + env->episode_length);
    }, [this](double latency) {
        if (trace.is_open()) {
            trace << env->episode_length << ',' << motion->frame << ',' << motion->frame * double(MotionClip::dt) << ',' << latency;
            const auto& root = env->robot->data.root_quat_w;
            trace << ',' << root.w() << ',' << root.x() << ',' << root.y() << ',' << root.z();
            for (float v : env->robot->data.root_ang_vel_b) trace << ',' << v;
            trace << ',' << init_quat.w() << ',' << init_quat.x() << ',' << init_quat.y() << ',' << init_quat.z();
            for (float q : env->robot->data.joint_pos) trace << ',' << q;
            for (float q : env->robot->data.joint_vel) trace << ',' << q;
            for (float q : env->action_manager->processed_actions()) trace << ',' << q;
            for (float a : env->action_manager->action()) trace << ',' << a;
            for (float o : env->last_observation.at("obs")) trace << ',' << o;
            trace << "\n";
        }
    }, window->frame_count, [this] {
        return guard.check(robot_quat_w(env.get()), motion_anchor_quat_w(motion), env->robot->data.root_quat_w,
                           env->robot->data.joint_pos, motion->joint_pos(), env->robot->data.joint_vel);
    });
}

void State_Mimic::run()
{
    auto action = worker.action();
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i)
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
}
