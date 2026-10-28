#include "FallGuard.h"
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool condition, const char* what) {
    if (!condition) throw std::runtime_error(std::string("Fall guard regression: ") + what);
}
int main() {
    const Eigen::Quaternionf upright = Eigen::Quaternionf::Identity();
    const Eigen::VectorXf zero = Eigen::VectorXf::Zero(29);
    FallGuard guard;
    require(guard.check(upright, upright, upright, zero, zero, zero).empty(), "upright on the reference");

    Eigen::Quaternionf tipped(Eigen::AngleAxisf(1.2f, Eigen::Vector3f::UnitX()));
    require(guard.check(upright, upright, tipped, zero, zero, zero) == "tilt", "pelvis tilt over 1 rad");
    Eigen::Quaternionf fallen(Eigen::AngleAxisf(1.6f, Eigen::Vector3f::UnitY()));
    require(guard.check(fallen, upright, upright, zero, zero, zero) == "anchor_gravity", "torso off the reference's tilt");
    require(guard.check(fallen, fallen, upright, zero, zero, zero).empty(), "a reference that leans as far");

    guard.reset();
    Eigen::VectorXf off = Eigen::VectorXf::Constant(29, 0.6f);
    for (int i = 0; i < 4; ++i)
        require(guard.check(upright, upright, upright, off, zero, zero).empty(), "joint error under 5 steps");
    require(guard.check(upright, upright, upright, off, zero, zero) == "joint_error", "joint error for 5 steps");

    guard.reset();
    Eigen::VectorXf fast = zero;
    fast[3] = 25.f;  // left knee, limit 20 rad/s
    require(guard.check(upright, upright, upright, zero, zero, fast).empty(), "one fast step");
    require(guard.check(upright, upright, upright, zero, zero, fast) == "joint_speed", "two fast steps");

    Eigen::VectorXf nan = zero;
    nan[0] = std::numeric_limits<float>::quiet_NaN();
    require(guard.check(upright, upright, upright, nan, zero, zero) == "non_finite_state", "NaN joints");

    YAML::Node cfg = YAML::Load("{tilt_rad: 1.5}");
    FallGuard loose(cfg);
    require(loose.check(upright, upright, tipped, zero, zero, zero).empty(), "configured tilt limit");
    std::cout << "fall guard ok\n";
}
