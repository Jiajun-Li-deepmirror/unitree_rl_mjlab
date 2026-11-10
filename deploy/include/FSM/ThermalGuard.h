#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "isaaclab/envs/manager_based_rl_env.h"

// Motor temperature (deg C): Go2-type motors report one value, hg-type (As2) two.
inline float motor_temperature(uint8_t t) { return static_cast<float>(t); }
inline float motor_temperature(const std::array<int16_t, 2>& t)
{
    return static_cast<float>(std::max(t[0], t[1]));
}

// Thermal guard of the leg motors while the policy runs.
//
// Standing and driving on wheels load the hip abduction motors without a break:
// the feet (wheels) sit outboard of the hip axis, so the body weight keeps pushing
// the legs apart, and on wheels the legs slowly splay further. Unitree's firmware
// cuts a motor at ~84-85 deg C (Go2-W, arXiv 2606.21387), and the robot then
// collapses. Before that the guard limits the speed (warn), stops the robot (stop:
// zero command, wheels braked) and lets it lie down through Passive (critical);
// each level clears once the hottest leg motor is `hysteresis` below its
// threshold. It also logs motor temperatures, hip angles and hip torques to CSV.
//
// With a legged policy trained to march in place (deploy.yaml `gait_march: true`,
// fifth round), the guard can make it step on the spot at the zero command
// (env->march keeps the gait clock running) instead of standing still, which
// loads one leg after the other: while the hottest leg motor is at `march_temp`
// or above (until it is `hysteresis` below), and with `march_idle_s` > 0 also for
// `march_idle_duration_s` after every `march_idle_s` of standing. Not while lying
// down (critical).
//
// Config (top-level `ThermalGuard` in config.yaml, all optional):
//   warn_temp 65, stop_temp 75, critical_temp 80 (deg C), hysteresis 5,
//   warn_speed_scale 0.5, leg_joints 12 (policy joints 0..n-1 are the legs),
//   hip_joints [0, 3, 6, 9] (policy order FL, FR, RL, RR), log_period 0.5 s,
//   log_dir "." (file thermal_<date>_<time>.csv; empty log_dir: no log),
//   march_temp (deg C; none: off), march_idle_s 0 (off), march_idle_duration_s 10.
class ThermalGuard
{
public:
    enum Level { kNormal = 0, kWarn = 1, kStop = 2, kCritical = 3 };

    ThermalGuard(const YAML::Node & cfg, isaaclab::ManagerBasedRLEnv* env)
    : _env(env), _start(std::chrono::steady_clock::now())
    {
        _thresholds = {
            0.0f,
            get<float>(cfg, "warn_temp", 65.0f),
            get<float>(cfg, "stop_temp", 75.0f),
            get<float>(cfg, "critical_temp", 80.0f),
        };
        _hysteresis = get<float>(cfg, "hysteresis", 5.0f);
        _warn_scale = get<float>(cfg, "warn_speed_scale", 0.5f);
        _leg_joints = get<int>(cfg, "leg_joints", 12);
        _hip_joints = get<std::vector<int>>(cfg, "hip_joints", std::vector<int>{0, 3, 6, 9});
        _log_period = get<float>(cfg, "log_period", 0.5f);
        _march_policy = env->cfg["gait_march"] && env->cfg["gait_march"].as<bool>();
        _march_temp = get<float>(cfg, "march_temp", std::numeric_limits<float>::infinity());
        _march_idle_s = get<float>(cfg, "march_idle_s", 0.0f);
        _march_idle_duration_s = get<float>(cfg, "march_idle_duration_s", 10.0f);
        const auto log_dir = get<std::string>(cfg, "log_dir", ".");
        if(!log_dir.empty()) _open_log(log_dir);
        spdlog::info("ThermalGuard: warn {} / stop {} / critical {} deg C (leg motors)",
                     _thresholds[kWarn], _thresholds[kStop], _thresholds[kCritical]);
        if(_march_policy)
            spdlog::info("ThermalGuard: march in place at the zero command from {} deg C, idle every {} s for {} s",
                         _march_temp, _march_idle_s, _march_idle_duration_s);
        else if(cfg && (cfg["march_temp"] || cfg["march_idle_s"]))
            spdlog::warn("ThermalGuard: march settings ignored, the policy is not trained to march (deploy.yaml gait_march)");
    }

    // Call every control cycle with the LowState subscription.
    template<class LowStatePtr>
    void update(const LowStatePtr & lowstate)
    {
        const auto & ids = _env->robot->data.joint_ids_map;
        const auto & motors = lowstate->msg_.motor_state();
        const int n_legs = std::min<int>(_leg_joints, ids.size());
        float hottest = 0.0f;
        int hottest_joint = -1;
        for(int i = 0; i < n_legs; ++i)
        {
            const float t = motor_temperature(motors[static_cast<int>(ids[i])].temperature());
            if(t > hottest) { hottest = t; hottest_joint = i; }
        }

        // Up at once; down one level at a time, `hysteresis` below the threshold.
        Level target = kNormal;
        for(int l = kCritical; l > kNormal; --l)
        {
            if(hottest >= _thresholds[l]) { target = static_cast<Level>(l); break; }
        }
        if(target > _level)
        {
            _level = target;
            spdlog::warn("ThermalGuard: leg motor {} (policy joint {}) at {:.0f} deg C -> {}",
                         hottest_joint >= 0 ? static_cast<int>(ids[hottest_joint]) : -1,
                         hottest_joint, hottest, _name(_level));
        }
        else if(_level > kNormal && hottest < _thresholds[_level] - _hysteresis)
        {
            _level = static_cast<Level>(_level - 1);
            spdlog::info("ThermalGuard: hottest leg motor {:.0f} deg C -> {}", hottest, _name(_level));
        }
        _env->command_scale = _level == kNormal ? 1.0f : _level == kWarn ? _warn_scale : 0.0f;
        if(_march_policy) _update_march(hottest);
        _log_row(lowstate, hottest);
    }

    bool critical() const { return _level >= kCritical; }

private:
    void _update_march(float hottest)
    {
        if(hottest >= _march_temp) _march_hot = true;
        else if(hottest < _march_temp - _hysteresis) _march_hot = false;
        // Policy steps the command has been zero (0 while moving).
        const float idle = _env->zero_command_time.load();
        bool idle_march = false;
        if(_march_idle_s > 0.0f && idle >= _march_idle_s)
        {
            const float cycle = _march_idle_s + _march_idle_duration_s;
            idle_march = std::fmod(idle - _march_idle_s, cycle) < _march_idle_duration_s;
        }
        const bool march = idle > 0.0f && _level < kCritical && (_march_hot || idle_march);
        if(march != _env->march.load())
            spdlog::info("ThermalGuard: {} marching in place (hottest leg motor {:.0f} deg C, idle {:.0f} s)",
                         march ? "start" : "stop", hottest, idle);
        _env->march = march;
    }

    template<class T>
    static T get(const YAML::Node & cfg, const char* key, const T & fallback)
    {
        return cfg && cfg[key] ? cfg[key].as<T>() : fallback;
    }

    static const char* _name(Level l)
    {
        switch(l)
        {
            case kWarn: return "warn: speed limited";
            case kStop: return "stop: zero command";
            case kCritical: return "critical: lying down (Passive)";
            default: return "normal";
        }
    }

    void _open_log(const std::string & dir)
    {
        const std::time_t now = std::time(nullptr);
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
        const std::string path = dir + "/thermal_" + stamp + ".csv";
        _log.open(path);
        if(!_log)
        {
            spdlog::warn("ThermalGuard: cannot write {}", path);
            return;
        }
        const int n = _env->robot->data.joint_ids_map.size();
        _log << "time_s,level,speed_scale,march,hottest_leg_c";
        for(int i = 0; i < n; ++i) _log << ",temp_" << i;
        for(int h : _hip_joints) _log << ",hip" << h << "_out_deg";
        for(int h : _hip_joints) _log << ",hip" << h << "_tau";
        _log << "\n";
        spdlog::info("ThermalGuard: logging to {}", path);
    }

    template<class LowStatePtr>
    void _log_row(const LowStatePtr & lowstate, float hottest)
    {
        if(!_log.is_open()) return;
        const float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - _start).count();
        if(t < _next_log) return;
        _next_log = t + _log_period;
        const auto & data = _env->robot->data;
        const auto & ids = data.joint_ids_map;
        const auto & motors = lowstate->msg_.motor_state();
        _log << t << "," << static_cast<int>(_level) << "," << _env->command_scale.load() << ","
             << static_cast<int>(_env->march.load()) << "," << hottest;
        for(size_t i = 0; i < ids.size(); ++i)
            _log << "," << motor_temperature(motors[static_cast<int>(ids[i])].temperature());
        // Hip abduction away from the body relative to the default pose (deg):
        // + for the left hips, - for the right ones (FL, RL at 0 and 6; FR, RR at 3, 9).
        for(int h : _hip_joints)
        {
            const float dq = motors[static_cast<int>(ids[h])].q() - data.default_joint_pos[h];
            const float side = (h / 3) % 2 == 0 ? 1.0f : -1.0f;
            _log << "," << side * dq * 57.29578f;
        }
        for(int h : _hip_joints) _log << "," << motors[static_cast<int>(ids[h])].tau_est();
        _log << "\n";
        _log.flush();
    }

    isaaclab::ManagerBasedRLEnv* _env;
    std::array<float, 4> _thresholds{};
    float _hysteresis = 5.0f;
    float _warn_scale = 0.5f;
    int _leg_joints = 12;
    std::vector<int> _hip_joints;
    float _log_period = 0.5f;
    float _next_log = 0.0f;
    Level _level = kNormal;
    bool _march_policy = false;
    float _march_temp = std::numeric_limits<float>::infinity();
    float _march_idle_s = 0.0f;
    float _march_idle_duration_s = 10.0f;
    bool _march_hot = false;
    std::chrono::steady_clock::time_point _start;
    std::ofstream _log;
};
