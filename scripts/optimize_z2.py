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
from vint_train.training.train_eval_loop import load_model as load_model_helper
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import os


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

    ema_model = None
    noise_scheduler = None
    # Support either single packaged ckpt or separate latest/optimizer/scheduler and train_config
    if cfg.get('ckpt'):
        ckpt_path = cfg.get('ckpt')
        ema_model = torch.load(ckpt_path, map_location=device)
        if isinstance(ema_model, dict) and 'ema_model' in ema_model:
            noise_scheduler = ema_model.get('noise_scheduler')
            ema_model = ema_model.get('ema_model')
    elif cfg.get('latest_pth'):
        # Need train_config to build model architecture
        train_cfg_path = cfg.get('train_config')
        assert train_cfg_path and os.path.exists(train_cfg_path), 'train_config path required to load latest_pth'
        with open(train_cfg_path, 'r') as f:
            train_cfg = yaml.safe_load(f)

        # Build model similarly to train/train.py
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
            # build vision encoder
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

        # load latest state dict
        latest_state = torch.load(cfg.get('latest_pth'), map_location='cpu')
        # train_eval_loop.load_model expects checkpoint dict for non-nomad, or state_dict for nomad
        if model_type == 'nomad':
            model.load_state_dict(latest_state, strict=False)
        else:
            # if latest_state is checkpoint dict with 'model' key
            if isinstance(latest_state, dict) and 'model' in latest_state:
                try:
                    loaded_model = latest_state['model']
                    state_dict = getattr(loaded_model, 'module', loaded_model).state_dict()
                except Exception:
                    state_dict = latest_state['model']
            else:
                state_dict = latest_state
            # strip module. prefix if present
            new_sd = {}
            for k, v in state_dict.items():
                new_k = k
                if k.startswith('module.'):
                    new_k = k[len('module.'):]
                new_sd[new_k] = v
            model.load_state_dict(new_sd, strict=False)

        ema_model = model
        # load optimizer/scheduler if provided
        if cfg.get('optimizer_pth'):
            optim_state = torch.load(cfg.get('optimizer_pth'), map_location='cpu')
        if cfg.get('scheduler_pth'):
            sched_state = torch.load(cfg.get('scheduler_pth'), map_location='cpu')


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
