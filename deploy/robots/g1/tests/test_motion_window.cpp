#include "MotionClip.h"
#include <iostream>
#include <limits>

static void require(bool condition) {
    if (!condition) throw std::runtime_error("Motion window regression");
}
int main(int argc, char** argv) {
    MotionWindow part(5.f, 7.f, 20.f);
    require(!part.finished(99, .02f));
    require(part.finished(100, .02f));
    require(std::abs(part.reference_time(50,.02f)-6.f)<1e-5f);
    require(part.reference_time(500,.02f)==7.f);
    MotionWindow jojo(0.f,52.38f,2619*.02f);
    require(jojo.frame_count==2619);
    require(!jojo.finished(2618,.02f));
    require(jojo.finished(2619,.02f));
    MotionWindow handshake(0.f,23.08f,1154*.02f);
    require(handshake.frame_count==1154);
    require(handshake.finished(1154,.02f));
    for (const auto& bounds : std::vector<std::pair<float,float>>{{-1,2},{2,2},{3,2},{0,21},{0,std::numeric_limits<float>::infinity()}}) {
        bool rejected=false;
        try { MotionWindow invalid(bounds.first,bounds.second,20); }
        catch (const std::runtime_error&) { rejected=true; }
        require(rejected);
    }
    if (argc > 1) {
        MotionClip motion(argv[1]);
        motion.update(1.0f);
        require(motion.frame==50);
        motion.update(motion.duration);
        require(motion.frame==motion.num_frames-1);
        require(motion.joint_pos().size()==29);
        for (int i=0;i<motion.num_frames;++i) {
            motion.update_frame(i);
            require(motion.frame==i);
        }
        std::cout << "Loaded " << motion.num_frames << " frames\n";
    }
    std::cout << "Motion window checks passed\n";
}
