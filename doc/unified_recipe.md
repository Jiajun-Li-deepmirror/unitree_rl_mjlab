# 统一配方（草案，2026-11-11）

配方由代码定义：`jobs/recipes/unified.py`。训练链现在仍读 `jobs/recipes/<机型>.sh`，这份草案确认之前不接入。

```
python3 jobs/recipes/unified.py go2w          # 某个机型的 T / M / P 开关和机型参数，每项带状态和理由
python3 jobs/recipes/unified.py go2w --sh     # 写成训练链用的变量
python3 jobs/recipes/unified.py --diff        # 六个机型：和现在的配方相比多了 / 少了哪些开关
```

## 1. 结构

| 层 | 内容 | 开关数（T/M + P 独有） |
|---|---|---|
| 共用 | 六个机型都用：平地技能、步态时钟、站立 / 原地踏步、负载、楼梯布局、楼梯直线、原地转；P 加楼梯稳定 | 29 + 4 |
| 轮式 | Go2W / As2W / B2W：轮子时钟、抬轮、驾驶姿态、一套约束、平地速度、刹车、慢弧线、踏步抬脚、停车锁姿；P 加楼梯质量 | 27 + 15 |
| 腿式 | Go2 / As2 / B2：踩边、起转、停车、偏航斜坡、碰底、约束；P 加落脚质量 | 11 + 8 |
| 机型 | 只放物理量：负载上限和位置、墙前停距、轮缩放和半径、腿刚度、锁腿时长、楼梯比例上限；B2W 另有 5 个只属于它的开关 | — |

- T 和 M 分成两组（现在内容相同，结构上已经分开，改 M 不必重训 T）。
- 步数上限统一：T ≤ 5000，M 再 ≤ 10000，P 再 ≤ 6000（是上限，每段第一次通过就停）。
- 约束倍数统一从 1 倍开始（以前 Go2、B2W 的初值是续训时留下的）。
- 状态：确定 62 个、需验证 37 个、删除 2 个；B2W 的 `legkp15`、`leghold20`、`stairs35`、`b2wcons` 改成机型参数或并入共用约束。

## 2. 每个机型的变化

| 机型 | T/M（现在 → 统一） | P（现在 → 统一） | 主要变化 |
|---|---|---|---|
| Go2W | 47 → 56 | 68 → 75 | T/M 加 `brakeside` `drcritic` `slowarc2`（As2W/B2W 有）、`restlock` `standtgt`（B2W 的停车锁姿）、`spinflip` `spinhold` `marchclear` `marchpose`（从 P 挪到 T/M）；P 加 `idlestep` `stairstandw`（As2W 有） |
| As2W | 54 → 56 | 73 → 75 | 加 `restlock` `standtgt` |
| B2W | 55 → 61 | 64 → 80 | 换用轮式共用约束（`wheelcons` 一组，原 `b2wcons` 待合并）；加平地速度三项、踏步两项、原地转两项；P 加楼梯稳定三项、楼梯朝向、每级只碰一次等 9 项；`legkp15` `leghold20` `stairs35` 改为机型参数 |
| Go2 | 36 → 40 | 49 → 52 | 加 `gaitw1`（只有 Go2 没有）、`stairshare` 挪到 T/M、`basetouch` `flatorient3`（As2 有）、`sidew6l`；去掉 `spintight` |
| As2 | 38 → 40 | 51 → 52 | 加 `legcons`（Go2 有）、`sidew6l`；`stairshare` 挪到 T/M；去掉 `edgecons`（`legcons` 已包含）、`spintight` |
| B2 | 34 → 40 | 50 → 52 | 加 `basetouch` `flatorient3` `legcons` `sidew6l`；`spinflip` `spinhold` 从 P 挪到 T/M；去掉 `edgecons` `spintight` |

`spinhold` 统一放进 T/M 以后，P 不再给价值网络加输入，这次 Go2W / B2 进 P 时的维度崩溃从根上消失。

## 3. 需要先改的代码（接入前）

1. B2W 的物理量写进机器人配置：腿刚度 kp 240 / kd 6（原 `legkp15`）、锁腿 2.0 s（原 `leghold20`，做成所有轮式都可设的参数）、楼梯比例上限 0.35（原 `stairs35`）。
2. 合并约束：`b2wcons` 的 back_drift、descent 两项指标并入轮式共用约束；`b2wdesc` `bdhold` 依赖 `b2wcons`，一起改。
3. 训练链读 T、M、P 三组（`BL_PILOT_T` / `BL_PILOT_M` 分开），`recipe_stage.sh` 认 T 和 M 两组。
4. 写在奖励里的机型常数改成按机型换算，例如带负载站姿惩罚按「负载 / 12 kg」放大（12 kg 是 Go2 的上限）。

## 4. 风险和待定

- **最大风险：T/M 里放多少开关。** 统一用的是「各机型做法取并集」，T/M 普遍多了几个开关。As2 rp3 的 M 完全不上楼，嫌疑正是谱系后期才加进 T/M 的开关（`feetedge`、停车类、`yawslew` 等）；而 B2 rp4 带着同样的开关能上楼。两种做法：
  - A（草案现状）：并集放进 T/M，最统一；
  - B（精简）：T/M 只放从零学走和学爬楼必需的，其余修正一律放 P，P 练得更久；更接近各谱系当初真正的训练顺序。
  建议先用冒烟测试把机制走通，再拿一台轮式（Go2W）和一台腿式（As2）各做一次 A/B，只跑到 M 能否上楼起势（Go2W 进入 M 后 2500–3000 步），再定。
- **B2W 变化最大**（多 16 个 P 开关，换约束）：它是唯一真正按原配方交付的 B2W，统一后要重新验。
- **验收的统一**（另一项工作，见对话里的清单）：原地转 p90 和文档中位数不一致、踏步复位 2° 和「2 秒」不一致、tinynav 航向 3° 没写进文档、B2W tinynav 负载位置 0.30 m 和别处 0.15 m 不一致、运气规则说明和代码不一致、负重站立结果可能没等跑完就判。门槛保持绝对量（只有负载上限按机型）。
