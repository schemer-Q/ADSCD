"""
两阶段意图训练脚本（独立文件，不修改原有训练脚本）。
用法示例：
  python train_two_stage.py --config configs/train_two_stage.yaml --ckpt path/to/model_ckpt.pth

说明：此脚本展示如何把 TwoStageIntentModel 的 z2 作为扩散模型条件传入训练流程。
此脚本尽量复用现有数据加载器与工具函数；在你的环境中可能需要根据具体 model factory 调整 `load_model_callable`。
"""
import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so imports like `vint_train` resolve
# Project root is three levels up from this script: train/vint_train/training
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vint_train.training.train_utils import (
    normalize_data,
    get_delta,
)

from vint_train.models.vint.two_stage_intent import TwoStageIntentModel
from vint_train.training.train_eval_loop import load_model as load_model_helper
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import os


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

    # 载入扩散模型 callable（支持单一 ckpt 或 separate latest/optimizer/scheduler + train_config）
    model_callable = None
    if args.ckpt:
        model_callable = load_model_callable(args.ckpt)
        model_callable.to(device)
    elif getattr(args, 'latest_pth', None):
        assert getattr(args, 'train_config', None) and os.path.exists(args.train_config), 'train_config required to load latest_pth'
        with open(args.train_config, 'r') as f:
            train_cfg = yaml.safe_load(f)
        # build model per train_cfg (reuse logic from train.py)
        model_type = train_cfg.get('model_type')
        if model_type == 'gnm':
            model = GNM(
                train_cfg['context_size'],
                train_cfg['len_traj_pred'],
                train_cfg['learn_angle'],
                train_cfg['obs_encoding_size'],
                train_cfg['goal_encoding_size'],
            )
        elif model_type == 'vint':
            model = ViNT(
                context_size=train_cfg['context_size'],
                len_traj_pred=train_cfg['len_traj_pred'],
                learn_angle=train_cfg['learn_angle'],
                obs_encoder=train_cfg['obs_encoder'],
                obs_encoding_size=train_cfg['obs_encoding_size'],
                late_fusion=train_cfg.get('late_fusion', False),
                mha_num_attention_heads=train_cfg.get('mha_num_attention_heads', 4),
                mha_num_attention_layers=train_cfg.get('mha_num_attention_layers', 4),
                mha_ff_dim_factor=train_cfg.get('mha_ff_dim_factor', 4),
            )
        elif model_type == 'nomad':
            if train_cfg['vision_encoder'] == 'nomad_vint':
                vision_encoder = NoMaD_ViNT(
                    obs_encoding_size=train_cfg['encoding_size'],
                    context_size=train_cfg['context_size'],
                    mha_num_attention_heads=train_cfg['mha_num_attention_heads'],
                    mha_num_attention_layers=train_cfg['mha_num_attention_layers'],
                    mha_ff_dim_factor=train_cfg['mha_ff_dim_factor'],
                )
                vision_encoder = replace_bn_with_gn(vision_encoder)
            elif train_cfg['vision_encoder'] == 'vit':
                vision_encoder = ViT(
                    obs_encoding_size=train_cfg['encoding_size'],
                    context_size=train_cfg['context_size'],
                    image_size=train_cfg['image_size'],
                    patch_size=train_cfg['patch_size'],
                    mha_num_attention_heads=train_cfg['mha_num_attention_heads'],
                    mha_num_attention_layers=train_cfg['mha_num_attention_layers'],
                )
                vision_encoder = replace_bn_with_gn(vision_encoder)
            else:
                raise ValueError('vision_encoder type not supported in loader')

            noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=train_cfg['encoding_size'],
                down_dims=train_cfg.get('down_dims'),
                cond_predict_scale=train_cfg.get('cond_predict_scale'),
            )
            dist_pred_network = DenseNetwork(embedding_dim=train_cfg['encoding_size'])
            z_dim = train_cfg.get('z_dim', 16)
            model = NoMaD(
                vision_encoder=vision_encoder,
                noise_pred_net=noise_pred_net,
                dist_pred_net=dist_pred_network,
                z_dim=z_dim,
            )
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=train_cfg.get('num_diffusion_iters', 1000),
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        else:
            raise ValueError('model_type not supported for loading latest_pth')

        latest_state = torch.load(args.latest_pth, map_location='cpu')
        if model_type == 'nomad':
            model.load_state_dict(latest_state, strict=False)
        else:
            if isinstance(latest_state, dict) and 'model' in latest_state:
                try:
                    loaded_model = latest_state['model']
                    state_dict = getattr(loaded_model, 'module', loaded_model).state_dict()
                except Exception:
                    state_dict = latest_state['model']
            else:
                state_dict = latest_state
            new_sd = {}
            for k, v in state_dict.items():
                new_k = k
                if k.startswith('module.'):
                    new_k = k[len('module.'):]
                new_sd[new_k] = v
            model.load_state_dict(new_sd, strict=False)

        model_callable = model.to(device)
        # optimizer/scheduler restoration can be added if needed using args.optimizer_pth / args.scheduler_pth

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
    parser.add_argument('--train_config', type=str, required=False, help='Path to original train config YAML (required if using latest_pth)')
    parser.add_argument('--latest_pth', type=str, required=False, help='Path to latest.pth (model state_dict)')
    parser.add_argument('--optimizer_pth', type=str, required=False, help='Path to optimizer_latest.pth')
    parser.add_argument('--scheduler_pth', type=str, required=False, help='Path to scheduler_latest.pth')
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
    # optional separate checkpoint files
    c.train_config = args.train_config
    c.latest_pth = args.latest_pth
    c.optimizer_pth = args.optimizer_pth
    c.scheduler_pth = args.scheduler_pth
    train_loop(c)


if __name__ == '__main__':
    main()
