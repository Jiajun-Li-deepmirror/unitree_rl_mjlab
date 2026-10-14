// Automated DDS integration runner. Loopback and a nonzero DDS domain are
// mandatory; this executable never publishes motor commands to a robot.
#include "unitree_sdk2_bridge.h"
#include <fstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <csignal>

static volatile std::sig_atomic_t interrupted = 0;
static void stop(int) { interrupted = 1; }

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: dance_headless scenario.yaml");
        auto cfg = YAML::LoadFile(argv[1]);
        int domain = cfg["domain_id"].as<int>();
        if (domain <= 0 || domain > 232) throw std::runtime_error("Simulation requires an isolated nonzero DDS domain");
        param::config.robot = "g1";
        param::config.robot_scene = cfg["scene"].as<std::string>();
        param::config.use_joystick = 0;
        param::config.print_scene_information = 0;
        param::config.enable_elastic_band = 0;
        char error[1024] = {};
        mjModel* model = mj_loadXML(param::config.robot_scene.c_str(), nullptr, error, sizeof(error));
        if (!model) throw std::runtime_error(error);
        if (model->nu != 29 || model->nq != 36 || model->nv != 35)
            throw std::runtime_error("Expected a floating-base G1 with 29 actuators");
        mjData* data = mj_makeData(model);
        model->opt.timestep = 0.002;
        mj_forward(model, data);
        unitree::robot::ChannelFactory::Instance()->Init(domain, "lo");
        G1Bridge bridge(model, data);
        const double duration = cfg["duration"].as<double>();
        const double support_until = cfg["support_until"].as<double>();
        const double dance_time = cfg["dance_time"].as<double>();
        const double abort_time = cfg["abort_time"].as<double>(-1);
        const double dropout_time = cfg["dropout_time"].as<double>(-1);
        const auto button = cfg["dance_button"].as<std::string>();
        if (button != "A" && button != "Y") throw std::runtime_error("Dance button must be A or Y");
        std::ofstream log(cfg["trace"].as<std::string>());
        if (!log) throw std::runtime_error("Cannot create simulation trace");
        log << "time,supported";
        for (int i=0; i<model->nq; ++i) log << ",qpos" << i;
        for (int i=0; i<model->nv; ++i) log << ",qvel" << i;
        for (int i=0; i<model->nu; ++i) log << ",torque" << i;
        log << '\n' << std::setprecision(9);
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        const auto began = std::chrono::steady_clock::now();
        auto next = began;
        bool connected = false;
        double t = 0;
        int step = 0;
        std::cout << "SIM_READY domain=" << domain << " interface=lo" << std::endl;
        while (!interrupted && t < duration) {
            if (!connected && !bridge.lowcmd->isTimeout()) {
                connected = true;
                std::cout << "SIM_CONTROLLER_CONNECTED" << std::endl;
            }
            if (!connected && std::chrono::steady_clock::now() - began > std::chrono::seconds(30))
                throw std::runtime_error("Controller did not connect within 30 seconds");
            unitree::common::REMOTE_DATA_RX remote{};
            remote.RF_RX.head[0] = 0xfe;
            remote.RF_RX.head[1] = 0xef;
            auto pulse = [t](double start, double length) { return t >= start && t < start + length; };
            // Trigger axes need to settle before the edge-triggered button.
            remote.RF_RX.btn.components.L2 = pulse(0.5, 0.5) || (abort_time >= 0 && pulse(abort_time, 0.5));
            remote.RF_RX.btn.components.up = pulse(0.7, 0.2);
            remote.RF_RX.btn.components.R2 = pulse(3.3, 0.5);
            remote.RF_RX.btn.components.R1 = pulse(dance_time - 0.2, 0.5);
            remote.RF_RX.btn.components.A = pulse(3.5, 0.2) || (button == "A" && pulse(dance_time, 0.2));
            remote.RF_RX.btn.components.Y = button == "Y" && pulse(dance_time, 0.2);
            remote.RF_RX.btn.components.B = abort_time >= 0 && pulse(abort_time + 0.2, 0.2);
            // Publisher copies this message under its own lock in run().
            if (bridge.lowstate->trylock()) {
                std::memcpy(bridge.lowstate->msg_.wireless_remote().data(), remote.buff, 40);
                bridge.lowstate->unlock();
            }
            bridge.publish_state = dropout_time < 0 || t < dropout_time;
            bridge.run();
            if (connected) {
                mj_step(model, data);
                if (t < support_until) {
                    // Startup suspension only; never used during dance evaluation.
                    std::copy(model->qpos0, model->qpos0 + 7, data->qpos);
                    std::fill(data->qvel, data->qvel + 6, 0.0);
                    mj_forward(model, data);
                }
                if (step % 10 == 0) {
                    log << t << ',' << (t < support_until);
                    for (int i=0; i<model->nq; ++i) log << ',' << data->qpos[i];
                    for (int i=0; i<model->nv; ++i) log << ',' << data->qvel[i];
                    for (int i=0; i<model->nu; ++i) log << ',' << data->actuator_force[i];
                    log << '\n';
                }
                t = (++step) * model->opt.timestep;
            }
            next += std::chrono::milliseconds(2);
            std::this_thread::sleep_until(next);
        }
        std::cout << "SIM_FINISHED time=" << t << " startup_support_until=" << support_until << std::endl;
        // DDS objects are destroyed before process exit. No physics thread is used.
        return interrupted ? 130 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Simulation failed: " << error.what() << std::endl;
        return 1;
    }
}
