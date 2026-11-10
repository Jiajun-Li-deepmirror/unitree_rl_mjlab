#pragma once

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <string>

// Has the robot left the states training kept it in? The tracking env ends an episode once the
// torso's tilt differs from the reference's by more than its anchor_ori threshold (0.8 on the
// gravity z component), or the torso or a hand or foot is 25 cm off in height, so the policy never
// learned what to do beyond that: after a fall it thrashes on the floor. Past this guard the state
// goes to Passive (damping). The robot measures the tilt (IMU and waist encoders) but not heights,
// so they are stood in for by the joints: their mean distance from the reference, and their speeds
// against the motor limits. scripts/sim2sim.py --guard runs the same checks (FALL_GUARD there).
class FallGuard {
public:
    explicit FallGuard(const YAML::Node& cfg = YAML::Node())
    {
        auto get = [&](const char* key, float fallback) { return cfg && cfg[key] ? cfg[key].as<float>() : fallback; };
        anchor_gravity_z = get("anchor_gravity_z", 0.8f);
        tilt_rad = get("tilt_rad", 1.0f);
        joint_error_rad = get("joint_error_rad", 0.5f);
        joint_error_steps = static_cast<int>(get("joint_error_steps", 5));
        joint_speed_ratio = get("joint_speed_ratio", 1.0f);
        joint_speed_steps = static_cast<int>(get("joint_speed_steps", 2));
    }

    void reset() { errors.clear(); fast = 0; }

    // torso, reference torso and root orientations in the world; joints in SDK order. The reason, or "".
    std::string check(const Eigen::Quaternionf& torso, const Eigen::Quaternionf& ref_torso,
                      const Eigen::Quaternionf& root, const Eigen::VectorXf& q, const Eigen::VectorXf& ref_q,
                      const Eigen::VectorXf& dq)
    {
        if (!q.allFinite() || !dq.allFinite()) return "non_finite_state";
        float up = torso.toRotationMatrix()(2, 2), ref_up = ref_torso.toRotationMatrix()(2, 2);
        if (std::abs(up - ref_up) > anchor_gravity_z) return "anchor_gravity";
        if (std::acos(std::clamp(root.toRotationMatrix()(2, 2), -1.f, 1.f)) > tilt_rad) return "tilt";
        errors.push_back((q - ref_q).cwiseAbs().mean());
        if (static_cast<int>(errors.size()) > joint_error_steps) errors.pop_front();
        if (static_cast<int>(errors.size()) == joint_error_steps &&
            *std::min_element(errors.begin(), errors.end()) > joint_error_rad)
            return "joint_error";
        bool over = false;
        for (int i = 0; i < dq.size() && i < static_cast<int>(speed_limits.size()); ++i)
            over |= std::abs(dq[i]) > joint_speed_ratio * speed_limits[i];
        fast = over ? fast + 1 : 0;
        if (fast >= joint_speed_steps) return "joint_speed";
        return "";
    }

private:
    // Motor speed limits (rad/s), SDK order: legs, waist, arms (validate_motion.py VELOCITY_LIMITS).
    static constexpr std::array<float, 29> speed_limits = {
        32, 20, 32, 20, 37, 37, 32, 20, 32, 20, 37, 37, 32, 37, 37,
        37, 37, 37, 37, 37, 22, 22, 37, 37, 37, 37, 37, 22, 22};
    float anchor_gravity_z, tilt_rad, joint_error_rad, joint_speed_ratio;
    int joint_error_steps, joint_speed_steps;
    std::deque<float> errors;
    int fast = 0;
};
