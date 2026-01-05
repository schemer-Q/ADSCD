# README B: 显式导航 / 对抗意图解耦 + 开关控制（Adversarial Intent Factorization）

## 1. 目标与动机

本实现验证一种 **显式意图解耦的对抗导航框架**，将：

* “去目标（navigation）”
* “对抗他人（adversarial interaction）”

视为两种**本质不同的决策意图**，并通过结构化潜变量与开关机制进行控制。

核心假设：

> 对抗行为不是持续存在的策略噪声，而是仅在交互条件满足时被激活的高层决策。

---

## 2. 方法概述

### 2.1 潜变量设计

* `z_nav`：导航意图潜变量（目标导向）
* `z_adv`：对抗意图潜变量（交互策略）
* `g ∈ {0,1}`：对抗开关（是否处于交互状态）

组合形式：

```
z = [ z_nav , g · z_adv ]
```

---

## 3. 架构组成

* **Navigation Encoder `E_nav`**
  输入：观测 + 目标
  输出：`z_nav`

* **Adversarial Encoder `E_adv`**
  输入：观测 + 他人状态（或交互特征）
  输出：`z_adv`

* **Interaction Gate `G`**
  输入：交互特征（距离、TTC 等）
  输出：`g`

* **Diffusion Trajectory Decoder `D_diff`**
  条件：`z_nav` + `g·z_adv`

---

## 4. 训练逻辑

### 4.1 数据要求

* 普通导航数据（无交互 / 弱交互）
* 含交互或对抗行为的数据（可合成）

### 4.2 训练策略

* `z_nav` 始终参与训练
* `z_adv` **仅在 g=1 的样本中激活**
* 强制模型在 `g=0` 时退化为纯导航策略

### 4.3 损失设计

* 基础 diffusion 去噪损失
* 对抗样本中：交互相关奖励 / 代价
* 可选：对 `z_adv` 的幅度或影响范围约束

---

## 5. 推理与对抗控制

### 5.1 推理流程

1. 根据当前场景计算交互条件 → `g`
2. 若 `g=0`：

   * 使用 `z_nav` 生成标准导航轨迹
3. 若 `g=1`：

   * 固定 `z_nav`
   * 调节或优化 `z_adv` 以生成不同对抗风格

### 5.2 对抗生成方式

* 扫描 `z_adv` 潜空间
* 梯度优化 `z_adv`
* 固定导航目标不变

---

## 6. Demo 实验建议

### 可行性验证（Stage 1）

* `g=0` vs `g=1` 行为对比
* 固定 `z_nav`，改变 `z_adv`

### 性能验证（Stage 2）

* 在相同起终点下：

  * 成功率保持情况
  * 对他人轨迹的影响程度

---

## 7. 方法优势与定位

**优势**：

* 语义清晰、可解释
* 对抗行为可控、可开关
* 适合系统性生成 worst-case 交互

**定位**：

* 不是学习“整体策略风格”
* 而是建模“条件性对抗决策”

---

## 8. 最小 Demo 实现（Minimal Viability Demo）

本 Demo 目标不是性能最优，而是验证以下三个**关键可行性假设**：

1. `g=0` 时，模型行为退化为稳定的目标导航
2. 固定 `z_nav`，调节 `z_adv` 可以显著改变交互行为
3. 激活对抗意图不会系统性破坏到达目标的能力

---

### 8.1 最小系统组成（必须模块）

**只保留 4 个核心模块**：

1. `E_nav`：导航意图编码器

   * 输入：当前观测 + 目标（如相对位姿）
   * 输出：`z_nav ∈ R^d`

2. `E_adv`：对抗意图编码器

   * 输入：当前观测 + 他人相对状态
   * 输出：`z_adv ∈ R^k`

3. `G`：规则式交互开关（不训练）

   * 输入：最近距离 / TTC
   * 输出：`g ∈ {0,1}`

4. `D_diff`：条件 diffusion 轨迹解码器

   * 条件：`[z_nav, g·z_adv]`

> ❗ 不需要参考轨迹、不需要 ACT、不需要 prior encoder

---

### 8.2 最小数据设定

* 单目标导航任务
* 单一动态障碍物（可 scripted）
* 两类数据：

  * **非交互样本**：他人距离 > d_safe → `g=0`
  * **交互样本**：他人切入路径 → `g=1`

数据可以：

* 全部仿真生成
* 不要求真实对抗最优，只要行为差异明显

---

### 8.3 训练配置（最小版）

* **阶段 1：纯导航预训练**

  * 强制 `g=0`
  * 只训练 `E_nav + D_diff`

* **阶段 2：对抗意图注入**

  * 冻结 `E_nav`
  * 启用 `E_adv`
  * 仅在 `g=1` 样本中拼接 `z_adv`

损失函数（最简）：

```
L = L_diffusion
```

> 不加 reward、不加博弈、不加 KL

---

### 8.4 推理与可视化方式

**固定同一起点与目标**，对比以下 3 种模式：

1. `g=0`（纯导航 baseline）
2. `g=1, z_adv = 0`
3. `g=1, z_adv ~ N(0, I)` 或扫描

观察：

* 是否仍能到达目标
* 对他人轨迹是否产生明显影响

---

### 8.5 最小成功判据（Success Criteria）

Demo 认为成功，只需满足：

* `z_adv` 改变 → 交互行为明显改变
* `z_nav` 固定 → 全局目标不漂移
* `g=0` 时对抗行为消失

> ❗ 不要求对抗“更聪明”，只要求**可控且解耦**

---

### 8.6 Demo 输出建议

* 同一场景下多条轨迹叠加图
* `g=0 / g=1` 行为对比 GIF
* 固定 `z_nav`、不同 `z_adv` 的 trajectory fan-out

---

### 8.7 Demo 之后可以自然扩展的方向

* 学习式 gate `G`
* 梯度优化 `z_adv`
* 多智能体对抗
* 与 ACT-style latent 的对比实验

## 9. 信息流图

                         ┌─────────────────────────┐
                         │       obs_img           │
                         └─────────┬───────────────┘
                                   │
                                   ▼
                        ┌───────────────────────┐
                        │   Shared Visual       │
                        │     Backbone (ViT)    │
                        └─────────┬─────────────┘
          ┌────────────────────────┼───────────────────────────┐
          │                        │                           │
          ▼                        ▼                           ▼
┌─────────────────-┐       ┌──────────────────────┐    ┌─────────────────────────┐
│ E_nav Head MLP   │       │ E_adv Head MLP       │    │  Diffusion Decoder      │
│ (z_nav)          │       │ (z_adv)              │    │  Conditional on z       │
│ Inputs:          │       │ Inputs:              │    │ Inputs: z_nav + g·z_adv │
│ - visual_feat    │       │ - visual_feat        │    │ + obs_lowdim + obs_img  │
│ - obs_lowdim     │       │ - obs_lowdim         │    └───────────┬─────────────┘
│ - goal_img       │       │ - other_state        │                │
└─────────┬────────┘       └─────────┬────────────┘                ▼
          │                          │                          ┌─────────────┐
          ▼                          ▼                          │ Predicted   │
      z_nav (mean, logvar)       z_adv (mean, logvar)           │ Trajectory  │
                                                                └─────────────┘
                                           ▲
                                           │
                                     g ∈ {0,1} (interaction switch)
                                     └─ multiplies z_adv before feeding to decoder

## 10. 补充说明

  ### Adversarial Intent Factorization: Minimal Demo

  #### 1. 项目概述

  本项目实现了对抗意图分离与层次化潜变量的最小可行 Demo，用于验证在导航任务中可控地生成对抗轨迹，同时保证导航目标不被破坏。

  核心设计思想：

  * **导航潜变量 (`z_nav`)**：编码自主导航意图。
  * **对抗潜变量 (`z_adv`)**：编码对抗意图，仅在交互场景激活。
  * **交互开关 (`g`)**：二值控制 `z_adv` 是否生效。
  * **Diffusion 解码器**：基于 `[z_nav, g * z_adv]` 和观测信息生成动作轨迹。

  #### 2. 架构设计

  ##### 编码器与共享 Backbone

  * 两个编码器共享视觉 backbone（如 ViT）提取图像特征。
  * 各自拥有独立的 MLP 头：

    * `z_nav_head(visual_feat, obs_lowdim, goal)`
    * `z_adv_head(visual_feat, obs_lowdim, other_state)`
  * Backbone 只处理视觉特征，低维信息（目标、对手状态等）由各自 MLP 处理。

  ##### 对手状态表征

  * **训练阶段**：由仿真环境提供低维状态向量（如相对位置和速度）。
  * **推理阶段**：

    * 仿真环境可直接读取。
    * 真实世界部署需要通过感知+跟踪+预测模块估计对手状态。

  ##### 开关机制（Mask）

  * `g` 在 latent 层控制 `z_adv` 是否参与 Diffusion decoder：

  ```python
  z_combined = concat(z_nav, g * z_adv)
  ```

  * `g=0`：z_adv 不参与，梯度不会回传 E_adv。
  * `g=1`：z_adv 参与，梯度回传 E_adv。

  ##### Loss 与梯度传递

  * Diffusion decoder 的 loss 对组合 latent 回传梯度：

    * z_nav 总是更新。
    * z_adv 仅在 `g=1` 样本中更新。

  ##### 数据流

  1. 输入观测图像 → **共享 backbone** → 提取视觉特征。
  2. E_nav MLP → 生成 z_nav。
  3. E_adv MLP → 生成 z_adv（包含对手状态）。
  4. z_combined = `[z_nav, g * z_adv]` → **Diffusion decoder** → 生成动作轨迹。
  5. Diffusion loss → 梯度回传 → 更新有效编码器。

  #### 3. 最小 Demo 使用说明

  ##### 依赖环境

  * Python >= 3.9
  * PyTorch >= 2.1
  * Hydra
  * OmegaConf
  * Matplotlib
  * Numpy

  ##### 运行 Demo

  ```bash
  python minimal_demo.py
  ```

  * 会生成三种场景：

    1. `g=0`：纯导航行为。
    2. `g=1`：交互场景，z_adv 激活。
    3. `g=1` + z_adv 优化：固定 z_nav，优化 z_adv。
  * 可视化结果保存至 `demo_results` 目录。

  ##### 输出说明

  * `comparison_g0_g1.png`：安全行为 vs 对抗行为对比。
  * `comparison_adv_opt.png`：对抗行为优化前后对比。
  * 控制台打印 latent 范数和 MSE 差异，验证 z_adv 的有效性。

  #### 4. 扩展方向

  * 使用感知+预测模块替代仿真对手状态，以部署于真实世界。
  * 可加入 learned gate `G`，根据场景自适应激活对抗意图。
  * 支持多智能体交互与更复杂的对抗策略优化。

  #### 5. 参考信息流图

  ```
                          obs_img
                              │
                              ▼
                  ┌─────────────────┐
                  │ Shared Backbone │
                  └─────────┬───────┘
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    E_nav Head MLP    E_adv Head MLP   Diffusion Decoder
    (z_nav)           (z_adv)          (z_combined = [z_nav, g * z_adv])
    Inputs:           Inputs:          + obs_lowdim + obs_img
    - visual_feat     - visual_feat
    - obs_lowdim      - obs_lowdim
    - goal_img        - other_state
  ```
