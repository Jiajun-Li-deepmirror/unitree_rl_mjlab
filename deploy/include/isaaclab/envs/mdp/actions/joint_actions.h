// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

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
    }
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
        _zero_when_command_zero = cfg["zero_when_command_zero"].as<bool>(false);
        _brake_below_speed = cfg["brake_below_speed"].as<float>(std::numeric_limits<float>::infinity());
    }

    void process_actions(std::vector<float> actions) override
    {
        JointAction::process_actions(actions);
        if(!_zero_when_command_zero) return;
        const auto & cmd = env->velocity_command;
        if(!std::all_of(cmd.begin(), cmd.end(), [](float v) { return v == 0.0f; })) return;
        const auto & joint_vel = env->robot->data.joint_vel;
        for(int i(0); i<_action_dim; ++i) {
            const int id = _joint_ids.empty() ? i : _joint_ids[i];
            if(std::abs(joint_vel[id]) >= _brake_below_speed) return;
        }
        std::fill(_processed_actions.begin(), _processed_actions.end(), 0.0f);
    }

private:
    bool _zero_when_command_zero = false;
    float _brake_below_speed = std::numeric_limits<float>::infinity();
};

REGISTER_ACTION(JointPositionAction);
REGISTER_ACTION(JointVelocityAction);

};