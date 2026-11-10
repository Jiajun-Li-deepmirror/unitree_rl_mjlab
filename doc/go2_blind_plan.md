# Go2 盲走：全向移动 + 上下楼梯 改动计划

> 这是最初的实现计划，保留作设计记录。实际的训练流程（后来加的奖励、map3 微调）、测试结果和部署说明见 [go2_blind.md](go2_blind.md)。

## Context

目标：训练一个**盲的**（只用本体感知）Go2 velocity policy，用仓库自带的 C++ deploy 部署到真机。
- 平地全向移动要有遥控器手感：前后、左右、原地转、组合运动。原地转要抬腿踏步，不能扭身体。
- 零指令时保持默认姿态，站着不动。
- 正向上下最高 18cm 的楼梯，可以带少量角速度。加分项：在楼梯中间停住站稳，允许身体倾斜。
- 已确定的参数：摇杆推满对应 vx [-1.0, 1.5] m/s、vy ±0.6 m/s、wz ±1.5 rad/s；台阶最高 18cm。

现有 Go2 配置里和目标冲突的地方（下面的改动都是针对这些）：
1. 所有 env 都开了 heading 模式，wz 被改写成 0.5×朝向误差（`velocity_env_cfg.py:170`、`velocity_command.py:104/255`），训练中没有"恒定角速度"指令；部署时 wz 却直接来自摇杆。
2. 朝向误差收敛后总指令 <0.1，训练切到站立模式：步态和抬脚奖励关闭、站姿惩罚打开、相位清零，但仍要求小角速度，于是学会扭身体。角速度跟踪的 std 是 0.71，小指令几乎没有梯度。
3. 配置用的是 mjlab 自带的 `UniformVelocityCommandCfg`（`velocity_env_cfg.py:27`），本仓库 `velocity_command.py:77` 里"小指令置零"的逻辑实际没生效。
4. 零指令只占 5%；`stand_still` 惩罚关节偏离默认姿态，和楼梯上站立冲突。
5. `track_linear_velocity` 含 2·vz² 项（`rewards.py:39`），会惩罚上下楼。
6. 抬脚相关奖励用世界坐标 z，楼梯上全部算错。
7. 任何非足端碰撞都终止 episode（`go2/env_cfgs.py:110`），下楼时小腿蹭台阶会被直接终止。
8. 部署端把摇杆值直接当速度并 clamp（`observations.h:118-120`），推满只能到 1.0。而且只把相位置零、不把指令置零，和训练不一致。

## 总体方案

新增两个 task，代码路径完全相同，只有地形不同：
- **`Unitree-Go2-Blind-Flat`（阶段 1，快速验证）**：地形只有平地，大约 1500–3000 iter。用来尽早验证遥控手感、原地转抬腿和零指令站立，调好奖励。**这个 checkpoint 不用于阶段 2 的 resume。**
- **`Unitree-Go2-Blind`（阶段 2，正式训练）**：平地 + 上楼 + 下楼 + 轻微起伏，从零开始训。
  - 不 resume 的原因：rsl_rl 的 `EmpiricalNormalization` 按累计样本数更新（`rsl_rl/modules/normalization.py:57-59`）。阶段 1 结束时已累计约 3e8 个样本，统计量基本冻结，楼梯的 height_scan 输入会被平地统计量归一化到几十个 σ。
  - 另外 resume 会恢复 `common_step_counter`（`mjlab/rl/runner.py:164-169`），楼梯一出现时，指令范围和跟踪 std 就已经在最终值，课程失效。
- 两个 task 的 `experiment_name` 都是 `go2_blind`，默认用 tensorboard。
- 不改变现有 `Unitree-Go2-Flat/Rough` 和其他机器人的行为：共享函数只加带默认值的可选参数，不就地修改 `GO2_ARTICULATION` 等共享对象（用 `dataclasses.replace`）。

## 改动清单

### 1. 指令 — `src/tasks/velocity/mdp/velocity_command.py`
新增 `TerrainAwareVelocityCommand(Cfg)`，继承**本仓库的** `UniformVelocityCommand`，复用它的小指令置零、viser 摇杆 GUI 和 debug 可视化。设置 `heading_command=False`、`ranges.heading=None`、`rel_standing_envs=0`。
- **出生列判断楼梯**：由 `env.scene.terrain.terrain_types` 加列到子地形的映射得到，映射按 `mjlab/terrains/terrain_generator.py::_generate_curriculum_terrains` 的 cumsum(proportions) 规则复现。这个判断只用于指令采样和出生朝向，不用于奖励（机器人走出楼梯区以后它就不准了）。
- **平地采样模式**：零指令 20% / 纯 vx 15% / 纯 vy 15% / 纯 wz 15% / 组合 35%。范围取 `cfg.ranges`，由 `commands_vel` 课程修改。
- **楼梯 env 采样**：零指令 25%，否则 vx ∈ [0.2, 0.8]、vy ∈ [-0.2, 0.2]、wz ∈ [-0.3, 0.3]。
- **重采样间隔**：楼梯 env 1.5–4 s（覆写 `_resample` 后重设 `time_left`），平地 2–5 s。
- **指令距离**：累计每个 env 本 episode 的指令距离 Σ‖v_cmd,xy‖·dt，在 `reset()` 里清零，供地形课程使用。已确认 reset 顺序是 curriculum → scene → events → obs → command，所以课程能读到清零前的值。

### 2. 奖励 — `src/tasks/velocity/mdp/rewards.py`
- `track_linear_velocity` 加参数 `z_weight=2.0`（默认值不变），blind 任务设为 0。
- 新增 `_uneven_terrain_mask(env, sensor_name="terrain_scan", half_extent=0.4, threshold=0.05)`：每步用 base 周围 ±0.4m 内 terrain_scan 命中点的高差判断"脚下不平"。奖励的地形区分都用它，不用出生列。
- 新增奖励函数：
  - `feet_swing_clearance`（类）：落地时计算 `peak_z − max(z_离地, z_落地)`，平方误差，目标 0.10m，不需要射线。它替代 `foot_clearance` 和世界系的 swing height，上下台阶时都正确。
  - `feet_stumble`：足端接触力满足 ‖F_xy‖ > 2|F_z| 时计数，用来惩罚脚尖踢到台阶立面。需要确认 `feet_ground_contact` 的力坐标系，必要时这个任务单独用 `global_frame=True` 的传感器副本。
  - `stand_still_even`：现有 `stand_still` 只在地面平整时生效。
  - `variable_posture_terrain_aware`（继承 `variable_posture`）：地面不平且站立时改用 `std_walking`，允许倾斜的站姿。
  - `standing_joint_vel`（零指令时惩罚 Σq̇²）、`standing_feet_contact`（零指令时奖励着地脚比例）。这两项和 `stand_still_even` 都再乘以 `exp(−(‖v_xy‖² + ω_z²)/0.1)`，被推之后的恢复步不受惩罚。
  - `body_orientation_terrain_aware`：地面不平时乘以 0.3。
  - `lin_vel_z_even`：地面平整时惩罚 vz²。
- 复用现有的 `self_collision_cost` 作为大腿、小腿碰地惩罚，传入新的腿-地形接触传感器即可。

### 3. 观测
- 新增 `foot_height_rel_base`（足端 z − base z，critic 用），替换世界系的 `foot_height`。
- **actor**：base_ang_vel、projected_gravity、command、phase、joint_pos、joint_vel、actions。不含 height_scan，组级 `history_length=10`（0.2s），共 47×10 = 470 维。已确认 mjlab 的历史布局（逐项拼接、旧帧在前、reset 回填）和部署端 `use_gym_history=false` 的布局一致，ONNX 输入名都是 `obs`。
- **critic**：actor 的各项（不带历史）+ base_lin_vel + height_scan + foot_height_rel_base + foot_air_time / foot_contact / foot_contact_forces。

### 4. 事件和域随机化 — 新文件 `src/tasks/velocity/mdp/events.py`（加入 `mdp/__init__.py`）
- **出生**：`reset_root_state_terrain_aware`。平地 env 的 yaw 均匀采样；楼梯 env 的 yaw = k·90° ± 15°。其余逻辑同 `mdp.reset_root_state_uniform`。
- **关节初始化**：`reset_robot_joints` 的位置偏移改为 ±0.1 rad，适应真机切换进 policy 时姿态不完全一致。
- **新增 DR**：
  - `dr.body_mass`：base 加 −1 ~ +2 kg（startup）。
  - `dr.pd_gains`：kp、kd 各乘 [0.85, 1.15]（startup）。
  - **延迟**：hip/thigh/calf 三组执行器分别外包一层 `DelayedActuatorCfg(delay_max_lag=4, delay_hold_prob=1.0)`，再加 reset 事件 `dr.sync_actuator_delays(lag_range=(0, 4))`，让每个 env 的三组延迟一致、episode 内固定在 0–20ms。
- 保留现有的摩擦、质心、编码器偏差、外力推。
- **碰撞**：这个任务单独复制一份 `CollisionCfg`，把小腿碰撞几何的 condim 从 1 改成 3。否则小腿碰到台阶边缘时没有摩擦，会直接滑开。

### 5. 课程 — `src/tasks/velocity/mdp/curriculums.py`
- **`terrain_levels_cmd_distance`**（替代 `terrain_levels_vel`，只在阶段 2 使用）：实际走过的距离 > 3.0m 就升级；小于 0.5×指令距离就降级。原来的版本只看最后一次指令，零指令的 env 永远不会降级。
- **`reward_params`**（按步数修改奖励的 params，写法类似现有的 `reward_weight`）：约 2000 iter 后，线速度跟踪 std 0.5→0.35，角速度 0.71→0.45。
- **`commands_vel` 阶段**：开始时 vx [-0.6, 1.0]、vy ±0.4、wz ±1.0；约 1500 iter 后放到 vx [-1.0, 1.5]、vy ±0.6、wz ±1.5。

### 6. 环境配置 — 新文件 `src/tasks/velocity/config/go2/blind_env_cfgs.py`
`unitree_go2_blind_env_cfg(play: bool, stairs: bool)`，在 `unitree_go2_rough_env_cfg()` 基础上修改：
- **地形 `GO2_BLIND_TERRAINS_CFG`**：
  - 8×8，10 行 × 20 列，`curriculum=True`，生成器 border_width 保持 20。
  - 平地 0.30、random_rough（起伏 2–6cm）0.14。
  - 下楼 `BoxPyramidStairsTerrainCfg`（出生在顶部平台）、上楼 `BoxInvertedPyramidStairsTerrainCfg`（出生在坑底），各两种踏面宽 0.25 / 0.32，每种 0.14。
  - 台阶 `step_height_range=(0.05, 0.18)`，`platform_width=2.0`，子地形 border 0.5，共 7–10 级。
  - `max_init_terrain_level=3`。
  - 阶段 1 只保留平地，但 generator 和 terrain_scan 不变，保证代码路径一致。
- **传感器**：新增 `leg_terrain_contact`（`ContactSensorCfg`，thigh/calf 碰撞几何 vs terrain，history 4）。
- **终止**：`illegal_contact` 只匹配 `base\d_collision` 和 `.*_hip_collision`（用正则）。
- **奖励表**（初始权重）：

  | 项 | 权重 | 备注 |
  |---|---|---|
  | track_linear_velocity | 1.0 | std 0.5→0.35，z_weight=0 |
  | track_angular_velocity | 1.0 | std 0.71→0.45 |
  | lin_vel_z_even | -2.0 | 新 |
  | body_orientation_terrain_aware | -1.0 | 不平 ×0.3 |
  | pose（terrain aware） | 1.0 | 沿用 Go2 的 std 表 |
  | foot_gait | 0.5 | trot，周期 0.6s |
  | feet_swing_clearance | -0.5 | 目标 0.10m |
  | feet_stumble | -0.5 | 新 |
  | foot_slip | -0.5 | 原来 -0.25 |
  | stand_still_even | -1.0 | |
  | standing_joint_vel | -0.02 | 新 |
  | standing_feet_contact | 0.5 | 新 |
  | leg_terrain_contact | -1.0 | 复用 self_collision_cost |
  | body_ang_vel / angular_momentum / is_terminated / joint_acc / joint_pos_limits / action_rate / soft_landing | 不变 | |

- **play 模式**：
  - 调用 rough 之后重新设置 `curriculum=True`、5 行 × 10 列。
  - `randomize_terrain` 要放在 events 的**第一个**。reset 事件按插入顺序执行，否则机器人会被放到旧的地块上。
  - `cfg.ranges` 直接设为最终范围（play 时不跑课程）。
  - 关闭观测噪声和外力推。

### 7. 注册和 RL 配置
- `src/tasks/velocity/config/go2/__init__.py`：注册两个 task，`runner_cls=VelocityOnPolicyRunner`（导出 ONNX）。
- `src/tasks/velocity/config/go2/rl_cfg.py`：加 `unitree_go2_blind_ppo_runner_cfg()`。`experiment_name="go2_blind"`，`logger="tensorboard"`，网络 (512, 256, 128)。阶段 1 `max_iterations=3000`，阶段 2 20000，都可以在命令行覆盖。

### 8. 部署
- `deploy/include/isaaclab/envs/mdp/observations/observations.h` 的 `velocity_commands`：在 deploy.yaml 的 `commands.base_velocity` 下新增可选项 `scale_to_range: true`。
  - 打开后摇杆 [-1, 1] 线性映射到 [min, max]（正向乘 max，负向乘 |min|）。
  - 三维指令的范数 < 0.1 时整体置零，和训练一致，也起到摇杆死区的作用。
  - 不设这一项时行为不变，其他机器人不受影响。
- 新增 `deploy/robots/go2/config/policy/velocity/blind_v0/params/deploy.yaml`：观测还是现有的 7 项，每项 `history_length: 10`；ranges 改为新速度；`scale_to_range: true`。
- `deploy/robots/go2/config/config.yaml`：
  - FixStand 的最终姿态改成 policy 的默认关节角（按 SDK 顺序 FR/FL/RR/RL，用 `joint_ids_map` 换算）。v0 的默认关节角相同，所以不受影响。
  - 训练后把 `policy.onnx` 拷到 `blind_v0/exported/`，`policy_dir` 指向 blind_v0。

## 训练命令
```bash
conda activate unitree_rl_mjlab
# 阶段 1：平地快速验证（不用于 resume）
python scripts/train.py Unitree-Go2-Blind-Flat --env.scene.num-envs=4096
# 阶段 2：完整任务从零训练
python scripts/train.py Unitree-Go2-Blind --env.scene.num-envs=4096
```

## 验证
1. `python scripts/list_envs.py` 能看到两个新 task。
2. 冒烟测试：两个 task 各用 256 个 env 跑 5 iter，不报错。打印确认：
   - actor 是 470 维。
   - 出生列的楼梯判断和地形一致。
   - 指令模式比例符合设定。
   - 楼梯 env 的出生朝向对齐台阶。
   - `_uneven_terrain_mask` 在台阶上为真、平地为假。
   - 三组执行器的延迟一致。
3. 阶段 1 训完后，用 viser play 加摇杆滑块检查：
   - 纯 wz：四条腿交替抬起（看 foot contact）。
   - 零指令：静止，被推后能恢复再站住。
   - 小摇杆量：有响应。
   - TensorBoard 里看 error_vel_xy / error_vel_yaw、slip、swing clearance。
   - 不满意就调奖励，重跑阶段 1。
4. 阶段 2：地形 level 曲线上升。在 play 里专门测楼梯：上下 18cm，中途把指令置零看能否站住。
5. 上真机前（可选）：用仓库的 `simulate/`（unitree_mujoco）和 deploy 程序做 sim2sim。需要先 `sudo apt install` 文档里的依赖，并编译 deploy。

## 待确认（本计划不改）
- Go2 默认髋关节角是 L −0.1 / R +0.1（`go2_constants.py`），站姿偏窄；Unitree 自己的 Go2 配置用的是相反的符号。宽一点的站姿对楼梯和侧移更稳。如果要改，部署端的 `default_joint_pos` 和 FixStand 也要一起改。可以先训一版看效果再决定。
