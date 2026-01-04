import torch
from torch import nn
from typing import Optional

from .vit import ViT


class Encoder1(nn.Module):
    """
    Encoder1: 将观测序列、目标图像与参考轨迹编码为高级意图 z1。
    输入约定：
      - obs_img: [B, 3*C, H, W] (与现有 ViT 接口相同，包含历史帧)
      - goal_img: [B, 3, H, W]
      - ref_traj: [B, T, D] 参考轨迹（位置/动作序列的数值表示）
    输出：z1 [B, z1_dim]
    """
    def __init__(self, z1_dim: int = 256, traj_feat_dim: int = 128, vit_kwargs: Optional[dict] = None):
        super().__init__()
        vit_kwargs = vit_kwargs or {}
        self.vit = ViT(**vit_kwargs)
        # map ViT output to a feature
        self.img_proj = nn.Sequential(
            nn.LayerNorm(vit_kwargs.get('obs_encoding_size', 512)),
            nn.Linear(vit_kwargs.get('obs_encoding_size', 512), z1_dim),
            nn.GELU(),
        )

        # simple MLP for reference trajectory
        self.traj_mlp = nn.Sequential(
            nn.Linear(traj_feat_dim, traj_feat_dim),
            nn.GELU(),
            nn.LayerNorm(traj_feat_dim),
        )

        self.fuse = nn.Sequential(
            nn.Linear(z1_dim + traj_feat_dim, z1_dim),
            nn.GELU(),
            nn.LayerNorm(z1_dim),
        )

    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor, ref_traj: torch.Tensor) -> torch.Tensor:
        # ViT expects obs_img and goal_img in the format used by existing code
        img_feat = self.vit(obs_img, goal_img, input_goal_mask=torch.zeros(obs_img.size(0), dtype=torch.int64, device=obs_img.device))
        img_z = self.img_proj(img_feat)

        # assume ref_traj shape [B, T, D] -> pool over time
        traj_pooled = ref_traj.mean(dim=1)
        traj_z = self.traj_mlp(traj_pooled)

        z1 = self.fuse(torch.cat([img_z, traj_z], dim=-1))
        return z1


class Encoder2(nn.Module):
    """
    Encoder2: 将 z1 与当前观测（单帧图像 / 当前上下文）映射为可执行意图 z2。
    输入：
      - z1: [B, z1_dim]
      - cur_obs_img: [B, 3*C', H, W] （当前观测，可与 Encoder1 的 obs_img 格式不同）
    输出：z2 [B, z2_dim]
    """
    def __init__(self, z1_dim: int = 256, z2_dim: int = 128, vit_kwargs: Optional[dict] = None):
        super().__init__()
        vit_kwargs = vit_kwargs or {}
        self.vit = ViT(**vit_kwargs)
        enc_size = vit_kwargs.get('obs_encoding_size', 512)

        self.obs_proj = nn.Sequential(
            nn.LayerNorm(enc_size),
            nn.Linear(enc_size, z2_dim),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(z1_dim + z2_dim, z2_dim),
            nn.GELU(),
            nn.LayerNorm(z2_dim),
            nn.Linear(z2_dim, z2_dim),
        )

    def forward(self, z1: torch.Tensor, cur_obs_img: torch.Tensor, goal_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        # goal_img optional — some setups encode both obs and goal
        if goal_img is None:
            # if no goal provided, reuse a zero tensor of appropriate size
            B = cur_obs_img.size(0)
            device = cur_obs_img.device
            goal_img = torch.zeros((B, 3, cur_obs_img.size(2), cur_obs_img.size(3)), device=device)

        obs_feat = self.vit(cur_obs_img, goal_img, input_goal_mask=torch.zeros(cur_obs_img.size(0), dtype=torch.int64, device=cur_obs_img.device))
        obs_z = self.obs_proj(obs_feat)

        z2 = self.fuse(torch.cat([z1, obs_z], dim=-1))
        return z2


class TwoStageIntentModel(nn.Module):
    """
    包装器：同时持有 Encoder1 和 Encoder2，提供联合编码和仅 Encoder2 推理接口。
    """
    def __init__(self, z1_dim: int = 256, z2_dim: int = 128, vit_kwargs: Optional[dict] = None):
        super().__init__()
        self.encoder1 = Encoder1(z1_dim=z1_dim, vit_kwargs=vit_kwargs)
        self.encoder2 = Encoder2(z1_dim=z1_dim, z2_dim=z2_dim, vit_kwargs=vit_kwargs)

    def encode_stage1(self, obs_img: torch.Tensor, goal_img: torch.Tensor, ref_traj: torch.Tensor) -> torch.Tensor:
        return self.encoder1(obs_img, goal_img, ref_traj)

    def encode_stage2(self, z1: torch.Tensor, cur_obs_img: torch.Tensor, goal_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder2(z1, cur_obs_img, goal_img)

    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor, ref_traj: torch.Tensor, cur_obs_img: torch.Tensor) -> torch.Tensor:
        z1 = self.encode_stage1(obs_img, goal_img, ref_traj)
        z2 = self.encode_stage2(z1, cur_obs_img, goal_img)
        return z1, z2


def init_z1_from_dataset(batch, model: TwoStageIntentModel, device=None):
    """Helper: 从数据 batch 中用 Encoder1 初始化 z1"""
    device = device or next(model.parameters()).device
    obs = batch['obs_img'].to(device)
    goal = batch['goal_img'].to(device)
    ref_traj = batch['ref_traj'].to(device)
    with torch.no_grad():
        z1 = model.encode_stage1(obs, goal, ref_traj)
    return z1


def optimize_z2_for_diffusion(model: TwoStageIntentModel, diffusion_model, fixed_z1: torch.Tensor, cur_obs_img: torch.Tensor, steps: int = 50, lr: float = 1e-2):
    """
    推理时用于对 z2 直接做梯度优化的示例流程：
      - fixed_z1: 可以是 Encoder1 输出的固定向量，也可以是随机/可学习的初始向量
      - diffusion_model: 外部扩散模型，需提供一个 loss_fn 或能将条件 z2 映射为轨迹并返回与目标的损失
    本函数只演示如何对 z2 优化（不包含 diffusion 内部细节）。
    """
    device = fixed_z1.device
    z2 = model.encode_stage2(fixed_z1, cur_obs_img.to(device))
    z2 = z2.clone().detach().requires_grad_(True)

    optim = torch.optim.Adam([z2], lr=lr)
    for _ in range(steps):
        optim.zero_grad()
        # diffusion_model 应实现一个接口：loss = diffusion_model.loss_given_z2(z2, ...)
        loss = diffusion_model.loss_given_z2(z2, cur_obs_img)
        loss.backward()
        optim.step()
    return z2.detach()
