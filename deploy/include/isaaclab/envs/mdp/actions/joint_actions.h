// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <atomic>
#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/action_manager.h"

namespace isaaclab
{

class JointAction : public ActionTerm
{
public:
    JointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :ActionTerm(cfg, env)
    {
        if(cfg["joint_ids"].IsNull()) {
            _action_dim = env->robot->data.joint_ids_map.size();
        } else {
            _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
            _action_dim = _joint_ids.size();
        }
        _raw_actions.resize(_action_dim, 0.0f);
        _processed_actions.resize(_action_dim, 0.0f);
        if(!cfg["scale"].IsNull()) {
            _scale = cfg["scale"].as<std::vector<float>>();
        }
        if(!cfg["offset"].IsNull()) {
            _offset = cfg["offset"].as<std::vector<float>>();
        }
        if(!cfg["clip"].IsNull()) {
            _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
        }
    }

    virtual void process_actions(std::vector<float> actions)
    {
        // TODO: modify action by joint_ids
        _raw_actions = actions;
        for(int i(0); i<_action_dim; ++i)
        {
            if(!_scale.empty()) {
                _processed_actions[i] = _raw_actions[i] * _scale[i];
            } else {
                _processed_actions[i] = _raw_actions[i];
            }
            if(!_offset.empty()) {
                _processed_actions[i] += _offset[i];
            }
        }
        if(!_clip.empty())
        {
            for(int i(0); i<_action_dim; ++i) {
                _processed_actions[i] = std::clamp(_processed_actions[i], _clip[i][0], _clip[i][1]);
            }
        }
    }


    int action_dim() 
    {
        return _action_dim;
    }

    std::vector<float> raw_actions() 
    {
        return _raw_actions;
    }
    
    std::vector<float> processed_actions() 
    {
        return _processed_actions;
    }

    void reset()
    {
        _raw_actions.assign(_action_dim, 0.0f);
    }

protected:
    int _action_dim;
    std::vector<int> _joint_ids;

    std::vector<float> _raw_actions;
    std::vector<float> _processed_actions;

    std::vector<float> _scale;
    std::vector<float> _offset;
    std::vector<std::vector<float> > _clip;
};


class JointPositionAction : public JointAction
{
public:
    JointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
        // Optional standstill hold (wheeled legs), as HeldJointPositionAction in
        // training: once the velocity command (after its deadzone) has been zero,
        // not marching in place, for hold_after_s and every joint of this term
        // turns slower than hold_speed (rad/s), the targets stay at that step's;
        // they follow the policy again on a command, a march or a joint faster
        // than hold_release_speed (rad/s), and hold again once calm.
        _hold_after_s = cfg["hold_after_s"].as<float>(std::numeric_limits<float>::infinity());
        _hold_speed = cfg["hold_speed"].as<float>(0.1f);
        _hold_release_speed = cfg["hold_release_speed"].as<float>(1.0f);
    }

    void process_actions(std::vector<float> actions) override
    {
        JointAction::process_actions(actions);
        if(!std::isfinite(_hold_after_s)) return;
        const auto & cmd = env->velocity_command;
        const bool zero = std::all_of(cmd.begin(), cmd.end(), [](float v) { return v == 0.0f; })
            && !env->march.load();
        _zero_time = zero ? _zero_time + env->step_dt : 0.0f;
        const auto & joint_vel = env->robot->data.joint_vel;
        float speed = 0.0f;
        for(int i(0); i<_action_dim; ++i) {
            speed = std::max(speed, std::abs(joint_vel[_joint_ids.empty() ? i : _joint_ids[i]]));
        }
        // Half a step of slack: summed float steps fall just short of whole seconds.
        const bool due = _zero_time >= _hold_after_s - 0.5f * env->step_dt;
        const bool engage = zero && !_held && due && speed < _hold_speed;
        _held = zero && ((_held && speed <= _hold_release_speed) || engage);
        if(engage) _hold_target = _processed_actions;
        if(_held) _processed_actions = _hold_target;
    }

    void reset() override
    {
        JointAction::reset();
        _held = false;
        _zero_time = 0.0f;
    }

private:
    float _hold_after_s = std::numeric_limits<float>::infinity();
    float _hold_speed = 0.1f;
    float _hold_release_speed = 1.0f;
    float _zero_time = 0.0f;
    bool _held = false;
    std::vector<float> _hold_target;
};

class JointVelocityAction : public JointAction
{
public:
    JointVelocityAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
        // Optional parking brake (wheels): zero velocity targets while the
        // velocity command (after its deadzone) is zero and every joint of this
        // term turns slower than brake_below_speed (rad/s), as in training.
        // With brake_hold_kp > 0 a braked joint also holds the angle it had when
        // the brake engaged: the robot's State_RLBase sends that angle as the
        // position target with this kp, as hold_stiffness does in training
        // (BrakedJointVelocityAction). Damping alone lets wheels roll away on
        // stairs and slopes. Optional, as in training: with brake_release_ratio
        // the speed limit is that times brake_below_speed once braked (a braked
        // joint lets go only when pushed) and, with brake_after_s, also once the
        // command has been zero that long (a policy holding its wheels just above
        // brake_below_speed on a riser otherwise keeps releasing the brake).
        // With brake_max_forward (m/s) the brake also works while |forward
        // command| is at most that (stepping sideways, turning in place), as
        // brake_max_forward in training.
        _zero_when_command_zero = cfg["zero_when_command_zero"].as<bool>(false);
        _brake_max_forward = cfg["brake_max_forward"].as<float>(-1.0f);
        _brake_below_speed = cfg["brake_below_speed"].as<float>(std::numeric_limits<float>::infinity());
        _brake_hold_kp = cfg["brake_hold_kp"].as<float>(0.0f);
        _brake_after_s = cfg["brake_after_s"].as<float>(std::numeric_limits<float>::infinity());
        _brake_release_ratio = cfg["brake_release_ratio"].as<float>(1.0f);
        _hold_pos.resize(_action_dim, 0.0f);
    }

    void process_actions(std::vector<float> actions) override
    {
        JointAction::process_actions(actions);
        const bool brake = _brake_engaged();
        if(brake) {
            std::fill(_processed_actions.begin(), _processed_actions.end(), 0.0f);
            if(!_braked) {
                const auto & joint_pos = env->robot->data.joint_pos;
                for(int i(0); i<_action_dim; ++i) {
                    _hold_pos[i] = joint_pos[_joint_ids.empty() ? i : _joint_ids[i]];
                }
            }
        }
        // Set after the hold angles: the control thread reads braked() first.
        _braked = brake;
    }

    void reset() override
    {
        JointAction::reset();
        _braked = false;
        _zero_time = 0.0f;
    }

    // Brake state for the motor commands (read by the control thread).
    bool braked() const { return _braked; }
    float brake_hold_kp() const { return _brake_hold_kp; }
    float hold_pos(int i) const { return _hold_pos[i]; }

private:
    // Called once per policy step (process_actions).
    bool _brake_engaged()
    {
        if(!_zero_when_command_zero) return false;
        const auto & cmd = env->velocity_command;
        const bool idle = _brake_max_forward >= 0.0f
            ? std::abs(cmd[0]) <= _brake_max_forward
            : std::all_of(cmd.begin(), cmd.end(), [](float v) { return v == 0.0f; });
        if(!idle) {
            _zero_time = 0.0f;
            return false;
        }
        _zero_time += env->step_dt;
        // Half a step of slack: summed float steps fall just short of whole seconds.
        const bool wide = _braked || _zero_time >= _brake_after_s - 0.5f * env->step_dt;
        const float limit = wide ? _brake_release_ratio * _brake_below_speed : _brake_below_speed;
        const auto & joint_vel = env->robot->data.joint_vel;
        for(int i(0); i<_action_dim; ++i) {
            const int id = _joint_ids.empty() ? i : _joint_ids[i];
            if(std::abs(joint_vel[id]) >= limit) return false;
        }
        return true;
    }

    bool _zero_when_command_zero = false;
    float _brake_max_forward = -1.0f;
    float _brake_below_speed = std::numeric_limits<float>::infinity();
    float _brake_hold_kp = 0.0f;
    float _brake_after_s = std::numeric_limits<float>::infinity();
    float _brake_release_ratio = 1.0f;
    float _zero_time = 0.0f;
    std::atomic<bool> _braked{false};
    std::vector<float> _hold_pos;
};

REGISTER_ACTION(JointPositionAction);
REGISTER_ACTION(JointVelocityAction);

};