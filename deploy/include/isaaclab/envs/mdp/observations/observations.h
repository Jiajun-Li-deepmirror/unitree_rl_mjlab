// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cmd_cfg = env->cfg["commands"]["base_velocity"];
    const auto cfg = cmd_cfg["ranges"];

    // Optional: map full stick deflection to the range limits (positive side to
    // max, negative side to |min|) and zero small commands, matching training.
    const bool scale_to_range = cmd_cfg["scale_to_range"].as<bool>(false);

    const float stick[3] = {joystick->ly(), -joystick->lx(), -joystick->rx()};
    const char* keys[3] = {"lin_vel_x", "lin_vel_y", "ang_vel_z"};
    // A safety guard may scale the command down (see command_scale).
    const float guard_scale = env->command_scale.load();
    for(int i = 0; i < 3; ++i)
    {
        const float lo = cfg[keys[i]][0].as<float>();
        const float hi = cfg[keys[i]][1].as<float>();
        float v = stick[i];
        if(scale_to_range) v = v >= 0.0f ? v * hi : v * -lo;
        obs[i] = std::clamp(v * guard_scale, lo, hi);
    }

    if(scale_to_range)
    {
        const float zero_threshold = cmd_cfg["zero_threshold"].as<float>(0.1f);
        const float norm = std::sqrt(obs[0] * obs[0] + obs[1] * obs[1] + obs[2] * obs[2]);
        if(norm <= zero_threshold) std::fill(obs.begin(), obs.end(), 0.0f);
    }

    env->velocity_command = obs;
    return obs;
}

// Heading the command asks for minus the base heading (rad, wrapped, + = turn
// left), as mdp.heading_error in training: the target starts at the IMU yaw after
// a reset and integrates the commanded yaw rate every control step, held within
// max_error of the base yaw. Must come after velocity_commands in the yaml, whose
// deadzoned command it integrates.
REGISTER_OBSERVATION(heading_error)
{
    const auto & q = env->robot->data.root_quat_w;
    const float yaw = std::atan2(2.0f * (q.w() * q.z() + q.x() * q.y()),
                                 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z()));
    const float max_error = params["max_error"].as<float>(0.5f);
    if(env->heading_settle > 0)
    {
        env->heading_target = yaw;
        env->heading_settle -= 1;
    }
    else
    {
        env->heading_target += env->velocity_command[2] * env->step_dt;
    }
    float error = std::atan2(std::sin(env->heading_target - yaw), std::cos(env->heading_target - yaw));
    error = std::clamp(error, -max_error, max_error);
    env->heading_target = yaw + error;
    return std::vector<float>{error};
}

// Seconds the deadzoned command (velocity_command) has been zero while not
// marching in place, over horizon_s and clamped to 1, as mdp.stand_time in
// training: the policy tells a stop's settling from standing on. Must come after
// velocity_commands in the yaml.
REGISTER_OBSERVATION(stand_time)
{
    const float horizon = params["horizon_s"].as<float>(2.0f);
    const auto & c = env->velocity_command;
    const float norm = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    const bool still = norm < 0.1f && !env->march.load();
    env->stand_time = still ? env->stand_time + env->step_dt : 0.0f;
    return std::vector<float>{std::min(env->stand_time / horizon, 1.0f)};
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    auto cmd = isaaclab::mdp::velocity_commands(env, params);
    // With ref_speed the period shortens above that commanded planar speed to
    // period * ref_speed / speed, as mdp.gait_clock in training (the stride stays
    // the same length at speed; slower commands and turning in place keep period).
    // With foot_xy (the feet's nominal [x, y] in the base frame) the speed is the
    // fastest foot's under the command, |(vx - wz y, vy + wz x)| (arcs).
    if (params["ref_speed"])
    {
        const float ref_speed = params["ref_speed"].as<float>();
        float speed = std::sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1]);
        if (params["foot_xy"])
        {
            speed = 0.0f;
            for (const auto& foot : params["foot_xy"])
            {
                const float vx = cmd[0] - cmd[2] * foot[1].as<float>();
                const float vy = cmd[1] + cmd[2] * foot[0].as<float>();
                speed = std::max(speed, std::sqrt(vx * vx + vy * vy));
            }
        }
        if (speed > ref_speed)
        {
            period *= ref_speed / speed;
        }
    }
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] +
        cmd[1] * cmd[1] +
        cmd[2] * cmd[2]
    );

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    // Zero while standing; still running while marching in place (env->march).
    // With the drivable limits (wheeled, as mdp.phase in training) also zero under
    // a command the wheels drive: the clock runs only where the robot steps.
    bool still = cmd_norm < 0.1f;
    if (params["drivable_max_lin_vel_y"] && params["drivable_max_ang_vel_z"])
    {
        still = still || (std::abs(cmd[1]) <= params["drivable_max_lin_vel_y"].as<float>() &&
                          std::abs(cmd[2]) <= params["drivable_max_ang_vel_z"].as<float>());
    }
    if (still && !env->march.load())
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }

    return obs;
}

}
}