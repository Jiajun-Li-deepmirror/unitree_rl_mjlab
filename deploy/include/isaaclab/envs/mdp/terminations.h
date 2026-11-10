#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return !data.allFinite() || std::acos(std::clamp(-data[2], -1.f, 1.f)) > limit_angle;
}

}
}
