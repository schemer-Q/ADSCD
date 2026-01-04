"""
两阶段意图训练脚本（独立文件，不修改原有训练脚本）。
用法示例：
  python train_two_stage.py --config configs/train_two_stage.yaml --ckpt path/to/model_ckpt.pth

说明：此脚本展示如何把 TwoStageIntentModel 的 z2 作为扩散模型条件传入训练流程。
此脚本尽量复用现有数据加载器与工具函数；在你的环境中可能需要根据具体 model factory 调整 `load_model_callable`。
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vint_train.training.train_utils import (
    normalize_data,
    get_delta,
)

from vint_train.models.vint.two_stage_intent import TwoStageIntentModel


def load_model_callable(ckpt_path: str):
    """载入现有的 diffusion model callable 对象。这里假设保存的是整个 model 对象，或提供一个工厂函数。
    用户可根据实际代码替换此函数以返回满足原项目 `model(func_name=..., ...)` 调用接口的对象。
    """
    obj = torch.load(ckpt_path, map_location='cpu')
    return obj


def train_loop(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化两阶段意图模型（仅用于生成 z1, z2）
    vit_kwargs = dict(obs_encoding_size=512, context_size=5, image_size=128, patch_size=16)
    two_stage = TwoStageIntentModel(z1_dim=256, z2_dim=128, vit_kwargs=vit_kwargs).to(device)

    # 载入扩散模型 callable（需符合原训练流程接口）
    model_callable = load_model_callable(args.ckpt)
    model_callable.to(device)

    # 数据加载：用户应提供与 nomad 相同格式的数据集（包含 ref_traj）
    # 这里仅示例如何在 batch 中使用 z1/z2；数据加载器的具体实现请替换为项目中的 DataLoader
    from vint_train.data.dataset import NomadDataset  # 假设存在
    train_dataset = NomadDataset(args.data_dir, split='train')
    dataloader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=4)

    optim = torch.optim.Adam(list(model_callable.parameters()) + list(two_stage.parameters()), lr=1e-4)

    for epoch in range(args.epochs):
        for batch in dataloader:
            # batch 应包含 obs_img, goal_img, actions, distance, goal_pos, action_mask, ref_traj
            obs_image, goal_image, actions, distance, goal_pos, dataset_idx, action_mask, ref_traj = batch

            obs_image = obs_image.to(device)
            goal_image = goal_image.to(device)
            actions = actions.to(device)
            ref_traj = ref_traj.to(device)

            # 生成 z1,z2
            z1 = two_stage.encode_stage1(obs_image, goal_image, ref_traj)
            # 例如使用最后一帧作为当前观测
            cur_obs = obs_image[:, -3:, :, :]
            z2 = two_stage.encode_stage2(z1, cur_obs, goal_img=goal_image)

            # 扩散训练与原项目一致，但将 z2 作为 latent_z 传入模型（或按需映射到 model 接口）
            # 下面仅示例调用 noise_pred_net 的方式，具体参数请与项目一致
            # 采样 timesteps, noisy actions 等步骤省略，直接计算一个示例 loss
            # 用户应将下面替换为完整的 diffusion 训练步骤

            # 简单示例损失：将 z2 通过 model 的某个 head，或把 z2 作为 latent_z 参与 noise_pred
            timesteps = torch.zeros(actions.size(0), dtype=torch.long, device=device)
            # noisy_action 及其他在实际训练中需根据 noise_scheduler 生成
            noisy_action = torch.randn_like(actions)

            noise_pred = model_callable(
                "noise_pred_net",
                sample=noisy_action,
                timestep=timesteps,
                global_cond=None,
                latent_z=z2,
                lambda_cfg=1.0,
            )

            # toy loss：MSE between noise_pred and zero noise (占位，替换为真实 diffusion loss)
            loss = nn.functional.mse_loss(noise_pred, torch.zeros_like(noise_pred))

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"Epoch {epoch} done, sample loss {loss.item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='YAML config with ckpt and options')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # merge cfg into args-like simple object
    class Cfg: pass
    c = Cfg()
    c.data_dir = cfg.get('data_dir')
    c.ckpt = cfg.get('ckpt')
    c.batch_size = cfg.get('batch_size', 16)
    c.epochs = cfg.get('epochs', 10)
    train_loop(c)


if __name__ == '__main__':
    main()
