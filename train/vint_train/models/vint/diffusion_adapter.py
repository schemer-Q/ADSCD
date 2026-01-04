"""
扩散适配器：提供一个 loss_given_z2 接口，推理脚本可用来对 z2 直接做梯度优化。
此处实现为示例：使用 ema_model 的 `noise_pred_net` 做一次快速采样并用 `dist_pred_net` 计算代理损失。
实际策略可替换为完整的扩散采样 + 目标损失。
"""
import torch


class DiffusionAdapter:
    def __init__(self, ema_model, noise_scheduler, action_stats=None):
        self.ema_model = ema_model
        self.noise_scheduler = noise_scheduler
        self.action_stats = action_stats

    def loss_given_z2(self, z2: torch.Tensor, cur_obs_img: torch.Tensor, target=None):
        """示例代理损失：
        - 使用 z2 作为 latent_z，运行一次 noise_pred_net 推断，
        - 将得到的 diffusion 输出转换为动作并由 dist_pred_net 评估距离预测，
        - loss = predicted_distance.mean() （希望距离越小越好）

        注意：此方法为近似/快速代理。对于更严格的对抗样本生成，请用完整扩散采样并计算目标任务损失。
        """
        device = z2.device
        B = z2.shape[0]
        # 生成一个随机 noisy action 作为占位
        # 实际上应该从噪声调度器 add_noise / step 得到完整轨迹
        sample = torch.randn((B, 10, 2), device=device)

        # 使用 ema_model 的 noise_pred_net（假设存在该接口）
        # 这里 timestep 取 0 作示例
        timesteps = torch.zeros(B, dtype=torch.long, device=device)
        noise_pred = self.ema_model(
            "noise_pred_net",
            sample=sample,
            timestep=timesteps,
            global_cond=None,
            latent_z=z2,
            lambda_cfg=1.0,
        )

        # 将 noise_pred 转为动作（简化）并传入 dist_pred_net
        # 这里使用 noise_pred 的均值作为代理动作编码
        pred_action = noise_pred.mean(dim=1)
        try:
            # 多数实现 dist_pred_net 接收 flattened obsgoal_cond - 这里仅示例调用
            dist_pred = self.ema_model("dist_pred_net", obsgoal_cond=pred_action)
            loss = dist_pred.mean()
        except Exception:
            # 回退：使用 L2 范数作为代理损失
            loss = (pred_action ** 2).mean()

        return loss
