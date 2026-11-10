# Source before running go2-depth jobs on this machine.
export PYTHONPATH=/root/gpufree-data/go2-depth/depth_fe
# PyTorch bundles cuDNN 9.20; the system one (9.8) on LD_LIBRARY_PATH breaks Conv2d.
export LD_LIBRARY_PATH=/root/gpufree-data/envs/g1-dance/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export MUJOCO_GL=egl
PY=/root/gpufree-data/envs/g1-dance/bin/python
