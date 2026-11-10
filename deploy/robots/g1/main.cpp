#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include <csignal>

static volatile std::sig_atomic_t stop_requested = 0;
static void request_stop(int) { stop_requested = 1; }

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        std::exit(1);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    FSMState::lowstate->set_timeout_ms(100);
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(vm["domain-id"].as<int>(), vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + A] to start controlling the robot.\n";
    std::cout << "Dance bindings are listed in the selected FSM config. [L2 + B] -> damping.\n";

    while (!stop_requested)
    {
        sleep(1);
    }
    fsm->request_passive();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    fsm.reset();
    
    return 0;
}
