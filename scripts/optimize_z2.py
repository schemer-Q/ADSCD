"""
推理脚本：固定或加载 z1，然后对 z2 做梯度优化以生成对抗轨迹。
示例流程：
  - 加载 TwoStageIntentModel + ema_model + noise_scheduler
  - 从数据集中初始化 z1（或加载预设 z1）
  - 使用 DiffusionAdapter.loss_given_z2 对 z2 优化
  - 优化后的 z2 可传回扩散模型进行最终采样
"""
import argparse
import yaml
import torch

from vint_train.models.vint.two_stage_intent import TwoStageIntentModel, init_z1_from_dataset, optimize_z2_for_diffusion
from vint_train.models.vint.diffusion_adapter import DiffusionAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='/root/private_data/latent_diffusion_policy/ADSCD/configs/two_stage_smoke.yaml',
        help='YAML config with ckpt and options (default: /root/private_data/latent_diffusion_policy/ADSCD/configs/two_stage_smoke.yaml)'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化两阶段模型（配置应与训练时一致）
    vit_kwargs = dict(obs_encoding_size=cfg.get('obs_encoding_size', 512), context_size=cfg.get('context_size', 5), image_size=cfg.get('image_size', 128), patch_size=cfg.get('patch_size', 16))
    two_stage = TwoStageIntentModel(z1_dim=cfg.get('z1_dim', 256), z2_dim=cfg.get('z2_dim', 128), vit_kwargs=vit_kwargs).to(device)

    # 加载扩散模型 / ema_model
    ckpt_path = cfg.get('ckpt')
    ema_model = torch.load(ckpt_path, map_location=device)
    noise_scheduler = None
    if isinstance(ema_model, dict) and 'ema_model' in ema_model:
        noise_scheduler = ema_model.get('noise_scheduler')
        ema_model = ema_model.get('ema_model')

    adapter = DiffusionAdapter(ema_model, noise_scheduler)

    # 获取一个 batch 来初始化 z1（此处假设用户提供 init_z1_from_dataset 支持的 batch）
    data_sample = cfg.get('data_sample')
    if data_sample:
        batch = torch.load(data_sample, map_location=device)
        z1 = init_z1_from_dataset(batch, two_stage, device=device)
    else:
        # 随机初始化 z1（可学习或从先验采样）
        z1 = torch.randn((1, cfg.get('z1_dim', 256)), device=device)

    # 当前观测占位（用户应提供真实图像）
    cur_obs = torch.randn((z1.shape[0], 3, cfg.get('image_size', 128), cfg.get('image_size', 128)), device=device)

    # 使用示例优化流程（optimize_z2_for_diffusion 在 two_stage 模块中已实现）
    z2_opt = optimize_z2_for_diffusion(two_stage, adapter, z1, cur_obs, steps=cfg.get('steps', 50), lr=cfg.get('lr', 1e-2))

    # 保存优化后的 z2
    out_path = cfg.get('out_z2', 'z2_optimized.pt')
    torch.save(z2_opt.cpu(), out_path)
    print(f'Saved optimized z2 -> {out_path}')


if __name__ == '__main__':
    main()
