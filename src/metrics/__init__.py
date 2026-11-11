"""The acceptance metrics, one implementation for every place that measures them (go2-depth 2026-11-12, doc/redesign.md:
"一份规格表，一套量法 ... 训练中的约束、mjlab 验收、tinynav、真机日志，都调用同一个指标函数").

  terrain.Grid      a height raster of a MuJoCo model's static boxes, its step edges and the "near stairs" bands
  stairs.LeggedStairs  per-foot contact state machines: riser kicks, slips, placement on the tread (mid / edge)

Plain numpy, vectorised over robots and feet, no project imports: tinynav-sim (one robot per run) and mjlab (many)
feed them the same per-step quantities. The definitions are tools/go2_depth/acceptance/accept_stairs.py's (the
tinynav acceptance since 2026-11-05), moved here unchanged.
"""
