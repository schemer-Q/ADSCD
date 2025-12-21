**Adversarial Distribution Shaping via Conditional Diffusion**

NoMaD/
├─ model.py / network.py       # 1. 修改 forward 接收 latent z，并拼接到条件向量 c
├─ train.py / train_policy.py  # 2. 训练阶段采样 z、classifier-free 训练、loss 计算
├─ sample.py / inference.py    # 3. 推理阶段 latent 初始化 + 可梯度优化 + CFG 修正
├─ encoder.py                  # 4. 原始 encoder 保留，latent z 与条件拼接即可
├─ config.yaml / args.py       # 5. 新增 z_dim, p_drop, lambda_cfg, latent_lr, latent_steps 等参数
└─ critic.py (可选)            # 6. 风险评估模块，用于梯度指导 latent z

**train.py**
# === train.py ===

for step in range(num_steps):
    # 1. 从数据集中采样
    s, a_seq, c_goal = sample_batch(D)

    # 2. 采样 diffusion timestep
    t = random.randint(1, T)

    # 3. 前向扩散，生成带噪动作
    epsilon = torch.randn_like(a_seq)
    x_t = sqrt_alpha[t] * a_seq + sqrt_1m_alpha[t] * epsilon

    # 4. 采样 latent z
    z = torch.randn(batch_size, z_dim).to(device)

    # 5. classifier-free 丢弃
    if random() < p_drop:
        z = None
        c_goal = None

    # 6. 模型预测噪声
    eps_hat = model(x_t, t, c_goal, z)   # forward 修改接收 z

    # 7. 计算扩散 loss
    L_diff = F.mse_loss(eps_hat, epsilon)

    # 8. 可选风险加权
    w = compute_risk_weight(a_seq, s)   # 可选：TTC 或碰撞 indicator
    L = (w * L_diff).mean()

    # 9. 反向传播
    optimizer.zero_grad()
    L.backward()
    optimizer.step()

**sample.py**
# === sample.py ===

# 1. 当前状态与目标图像编码
c_goal = encoder(obs, goal_image)

# 2. 初始化 latent z
z = torch.randn(1, z_dim).to(device)
z.requires_grad = True  # 如果要做梯度优化

# 3. 可选 latent 优化（梯度上升）
for step in range(latent_steps):
    a_pred = diffusion_sample(s, c_goal, z)  # 反向扩散采样
    risk = R_critic(a_pred, s)
    grad_z = torch.autograd.grad(risk, z)[0]
    z = z + latent_lr * grad_z

z_adv = z.detach()  # 优化完成，固定 z

# 4. 反向扩散生成动作序列（带 CFG）
x_t = torch.randn(1, action_dim, T)  # 初始化噪声
for t in reversed(range(1, T+1)):
    eps_uncond = model(x_t, t, c_goal, None)
    eps_cond   = model(x_t, t, c_goal, z_adv)

    # CFG 修正
    eps_hat = eps_uncond + lambda_cfg * (eps_cond - eps_uncond)

    # Denoise step
    x_t = denoise_step(x_t, eps_hat, t)

a_adv = x_t  # 输出动作序列

# [关键点解释](image.png)
# [与直接修改goal引导碰撞的思路对比](image-1.png)