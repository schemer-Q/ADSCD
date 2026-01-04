# Two-Stage Intent README

本文档说明如何运行新增的两阶段意图训练与推理脚本，及所需的数据与 checkpoint 格式。

文件位置（参考）：
- 两阶段编码实现：[train/vint_train/models/vint/two_stage_intent.py](train/vint_train/models/vint/two_stage_intent.py)
- 扩散适配器：[train/vint_train/models/vint/diffusion_adapter.py](train/vint_train/models/vint/diffusion_adapter.py)
- 新训练脚本（独立）：[train/vint_train/training/train_two_stage.py](train/vint_train/training/train_two_stage.py)
- 推理脚本（z2 优化）：[scripts/optimize_z2.py](scripts/optimize_z2.py)

快速运行指南

1) 准备数据
- 数据集须兼容 nomad 的格式（dataloader 输出与现有训练一致）。新增要求：每个 batch 需包含 `ref_traj` 字段，形状为 `[B, T, D]`（参考轨迹的时序特征）。
- 训练/推理所需最小字段：`obs_img`, `goal_img`, `actions`, `distance`, `goal_pos`, `dataset_idx`, `action_mask`, `ref_traj`。

2) 准备 checkpoint
- `train_two_stage.py` 和 `optimize_z2.py` 假设可以通过 `torch.load(ckpt_path)` 得到一个模型对象或包含模型与 scheduler 的 dict。
- 推荐格式：保存一个 dict：`{"ema_model": ema_model, "noise_scheduler": noise_scheduler}`，或直接保存可调用的模型对象（项目现有训练代码如何保存即用其格式）。
- `ema_model` 需支持调用接口示例：
  - `ema_model("noise_pred_net", sample=..., timestep=..., global_cond=..., latent_z=..., lambda_cfg=...)`
  - `ema_model("dist_pred_net", obsgoal_cond=...)`
  如果你的实现接口不同，请在 `diffusion_adapter.py` 与训练脚本中做小的适配。

3) 训练示例
运行独立训练示例（仅示例，需根据项目替换 model loader 与 dataset）：
```bash
python train/vint_train/training/train_two_stage.py --data_dir /path/to/data --ckpt /path/to/ckpt.pth --batch_size 16 --epochs 10
```

说明：`train_two_stage.py` 演示把 `TwoStageIntentModel` 产出的 `z2` 作为扩散模型的 `latent_z` 输入。脚本中有注释提示哪些部分需用你项目的 model factory/noise scheduler 替换以完成真实训练流程。

4) 推理 / z2 优化示例
用 `optimize_z2.py` 固定或初始化 `z1` 并对 `z2` 做梯度优化：
```bash
python scripts/optimize_z2.py --ckpt /path/to/ema_ckpt.pth --data_sample /path/to/sample_batch.pt --steps 100 --lr 1e-2
```
- 如果不提供 `--data_sample`，脚本会随机初始化 `z1`。
- 输出：优化后的 `z2` 保存在 `z2_optimized.pt`。

实现说明与性能注意
- 两阶段好处：意图分层、搜索空间更小、可用性强。挑战是需要参考轨迹数据，且在推理阶段对 `z2` 做梯度优化会比较耗时，不适合严格的实时场景。
- `DiffusionAdapter.loss_given_z2` 当前实现为代理/快速版本（不执行完整的扩散采样），便于快速迭代；要生成严格意义上的对抗轨迹，请在该文件中实现完整扩散采样 + 任务损失计算（会显著增加计算开销）。

调整点（按需）
- 若你的模型工厂/保存格式与示例不符，请在 `train_two_stage.py` 中替换 `load_model_callable`，并在 `diffusion_adapter.py` 中修改 `loss_given_z2` 以匹配你的 `ema_model` 接口。
- 若数据 loader 名称不同，请替换脚本中的 `NomadDataset` 导入与构造。

如需我把示例脚本直接对接到你现有的 model checkpoint（做一次 smoke 测试），请提供可用的 checkpoint 路径或说明项目中如何通过代码实例化/加载原始模型与 noise scheduler，我会把脚本中的占位加载逻辑替换为真实加载并跑一次快速检查。
