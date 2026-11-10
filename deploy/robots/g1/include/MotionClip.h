#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cnpy.h>
#include <Eigen/Dense>

// The exported tracking policies use SDK joint order, float32 motion arrays,
// wxyz root quaternions and a 50 Hz reference clock.
class MotionClip
{
public:
    explicit MotionClip(const std::string& path)
    {
        auto arrays = cnpy::npz_load(path);
        auto get = [&](const char* name, size_t rank, size_t last) -> const cnpy::NpyArray& {
            auto it = arrays.find(name);
            if (it == arrays.end()) throw std::runtime_error(std::string("Missing motion field: ") + name);
            const auto& a = it->second;
            if (a.word_size != sizeof(float) || a.fortran_order || a.shape.size() != rank ||
                a.shape.back() != last || a.shape.front() == 0)
                throw std::runtime_error(std::string("Invalid float32 motion shape: ") + name);
            for (size_t i = 0; i < a.num_vals; ++i)
                if (!std::isfinite(a.data<float>()[i])) throw std::runtime_error("Non-finite motion data");
            return a;
        };
        const auto& q = get("joint_pos", 2, 29);
        const auto& dq = get("joint_vel", 2, 29);
        const auto& pos = get("body_pos_w", 3, 3);
        const auto& quat = get("body_quat_w", 3, 4);
        num_frames = q.shape[0];
        if (dq.shape[0] != q.shape[0] || pos.shape[0] != q.shape[0] ||
            quat.shape[0] != q.shape[0] || pos.shape[1] == 0 || pos.shape[1] != quat.shape[1])
            throw std::runtime_error("Motion frame/body counts do not match");
        duration = num_frames * dt;
        for (int i = 0; i < num_frames; ++i) {
            positions.emplace_back(Eigen::Vector3f::Map(pos.data<float>() + i * pos.shape[1] * 3));
            const float* v = quat.data<float>() + i * quat.shape[1] * 4;
            Eigen::Quaternionf rotation(v[0], v[1], v[2], v[3]);
            if (std::abs(rotation.norm() - 1.f) > 0.01f) throw std::runtime_error("Invalid motion quaternion");
            rotations.push_back(rotation.normalized());
            qs.emplace_back(Eigen::VectorXf::Map(q.data<float>() + i * 29, 29));
            dqs.emplace_back(Eigen::VectorXf::Map(dq.data<float>() + i * 29, 29));
        }
    }

    void update(float time) {
        update_frame(static_cast<long>(std::floor(double(std::max(time, 0.f)) / dt + 1e-3)));
    }
    void update_frame(long value) { frame = static_cast<int>(std::clamp(value, 0L, long(num_frames - 1))); }
    Eigen::Vector3f root_position() const { return positions[frame]; }
    Eigen::Quaternionf root_quaternion() const { return rotations[frame]; }
    Eigen::VectorXf joint_pos() const { return qs[frame]; }
    Eigen::VectorXf joint_vel() const { return dqs[frame]; }
    static constexpr float dt = 0.02f;
    int frame = 0, num_frames = 0;
    float duration = 0;
private:
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Quaternionf> rotations;
    std::vector<Eigen::VectorXf> qs, dqs;
};

struct MotionWindow {
    float start, end;
    long first_frame, frame_count;
    MotionWindow(float start_, float end_, float duration) : start(start_), end(end_) {
        if (!std::isfinite(start) || !std::isfinite(end) || start < 0 || end <= start || end > duration + 1e-5f)
            throw std::runtime_error("Require 0 <= time_start < time_end <= motion duration");
        first_frame = std::lround(double(start) / MotionClip::dt);
        const long last_frame = std::lround(double(end) / MotionClip::dt);
        if (std::abs(first_frame * double(MotionClip::dt) - start) > 1e-4 ||
            std::abs(last_frame * double(MotionClip::dt) - end) > 1e-4)
            throw std::runtime_error("Motion window must align with the 50 Hz frame grid");
        frame_count = last_frame - first_frame;
    }
    bool finished(long steps, float dt) const { return steps >= frame_count; }
    float reference_time(long steps, float dt) const { return std::min(start + steps * dt, end); }
};
