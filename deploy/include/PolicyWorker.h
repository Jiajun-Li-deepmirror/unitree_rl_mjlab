#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <limits>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/terminations.h"

// Only the worker touches the environment once started. The 1 kHz sender
// reads a complete action snapshot, never an action being overwritten.
class PolicyWorker {
public:
    ~PolicyWorker() { stop(); }
    void stop() {
        running = false;
        if (thread.joinable()) thread.join();
    }
    void start(isaaclab::ManagerBasedRLEnv* env, std::function<void()> before_step = {},
               std::function<void(double)> after_step = {}, long max_steps = std::numeric_limits<long>::max()) {
        stop();
        fault = false;
        steps = 0;
        last_update = now();
        {
            std::lock_guard<std::mutex> lock(mutex);
            command.assign(env->robot->data.joint_pos.data(),
                           env->robot->data.joint_pos.data() + env->robot->data.joint_pos.size());
        }
        running = true;
        thread = std::thread([this, env, before_step, after_step, max_steps] {
            const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(env->step_dt));
            auto next = std::chrono::steady_clock::now();
            while (running) {
                try {
                    auto began = std::chrono::steady_clock::now();
                    if (before_step) before_step();
                    env->step();
                    auto action = env->action_manager->processed_actions();
                    if (action.size() != env->robot->data.joint_ids_map.size() ||
                        !std::all_of(action.begin(), action.end(), [](float x) { return std::isfinite(x); }))
                        throw std::runtime_error("Invalid policy action");
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        command = std::move(action);
                    }
                    if (isaaclab::mdp::bad_orientation(env, 1.0)) fault = true;
                    last_update = now();
                    if (after_step) after_step(std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
                    next += period;
                    // Do not issue a burst of stale reference frames after a long stall.
                    if (std::chrono::steady_clock::now() - next > period * 4) fault = true;
                    if (fault) break;
                    std::this_thread::sleep_until(next);
                    // Count a frame only after its action has been held for dt.
                    steps = env->episode_length;
                    if (env->episode_length >= max_steps) break;
                } catch (const std::exception& error) {
                    spdlog::error("Policy stopped: {}", error.what());
                    fault = true;
                    break;
                }
            }
        });
    }
    std::vector<float> action() {
        std::lock_guard<std::mutex> lock(mutex);
        return command;
    }
    bool unhealthy() const { return fault || now() - last_update.load() > 0.1; }
    std::atomic<long> steps{0};
private:
    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    std::atomic<bool> running{false}, fault{false};
    std::atomic<double> last_update{0};
    std::thread thread;
    std::mutex mutex;
    std::vector<float> command;
};
