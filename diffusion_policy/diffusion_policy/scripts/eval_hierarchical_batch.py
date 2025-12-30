#!/usr/bin/env python3
"""Batch evaluator for hierarchical diffusion policy.

Generates dummy test cases (same style as demo_hierarchical_diffusion.py),
runs multiple trials across modes (adv disabled / enabled / optimized),
computes MSE against ground truth, measures inference/opt time,
saves per-run CSV and trajectory plots.

Usage:
  python diffusion_policy/diffusion_policy/scripts/eval_hierarchical_batch.py --n-runs 50 --output-dir demo_eval
"""
import argparse
import time
import pathlib
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
from diffusion_policy.model.common.normalizer import LinearNormalizer


def visualize_trajectory(gt_action, pred_action, out_path, title):
    if torch.is_tensor(gt_action):
        gt_action = gt_action.cpu().numpy()
    if torch.is_tensor(pred_action):
        pred_action = pred_action.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10,4))
    n_actions = gt_action.shape[-1]
    for i in range(n_actions):
        ax.plot(gt_action[:, i], label=f'GT {i+1}', alpha=0.9)
        ax.plot(pred_action[:, i], linestyle='--', label=f'Pred {i+1}', alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel('Timestep')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def align_and_mse(gt, pred):
    gt_np = gt if isinstance(gt, np.ndarray) else gt.cpu().numpy()
    pred_np = pred if isinstance(pred, np.ndarray) else pred.cpu().numpy()
    L = min(gt_np.shape[0], pred_np.shape[0])
    gt_a = gt_np[:L]
    pred_a = pred_np[:L]
    mse = float(np.mean((gt_a - pred_a) ** 2))
    return mse, gt_a, pred_a


def main(args):
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load hydra config (same location as demo)
    current_file = pathlib.Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    config_path = current_file.parent.parent / 'config'
    cfg_name = 'train_diffusion_hierarchical_workspace'

    with hydra.initialize_config_dir(config_dir=str(config_path)):
        cfg = hydra.compose(config_name=cfg_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate policy
    policy = hydra.utils.instantiate(cfg.policy)
    policy.to(device)
    policy.eval()

    # create normalizer (fit on random dummy data matching shapes)
    normalizer = LinearNormalizer()
    batch_size = 2
    dummy_batch = {
        'obs': torch.randn(batch_size, cfg.horizon, cfg.obs_dim),
        'action': torch.randn(batch_size, cfg.horizon, cfg.action_dim)
    }
    normalizer.fit(dummy_batch)
    policy.set_normalizer(normalizer)

    # prepare CSV
    csv_path = out_dir / 'results.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['run', 'seed', 'mode', 'mse', 'gt_len', 'pred_len', 'inf_time_s', 'opt_time_s'])

    modes = ['disabled', 'enabled', 'optimized']

    rng = np.random.RandomState(args.seed)

    for run in range(args.n_runs):
        seed = int(rng.randint(0, 2**31 - 1))
        torch.manual_seed(seed)

        # generate dummy data (same as demo)
        context_size = cfg.vit_encoder.context_size
        obs_img = torch.randn(batch_size, 3*context_size, 128, 128, device=device)
        goal_img = torch.randn(batch_size, 3, 128, 128, device=device)
        obs = torch.randn(batch_size, cfg.horizon, cfg.obs_dim, device=device)
        action = torch.randn(batch_size, cfg.horizon, cfg.action_dim, device=device)  # ground truth

        # per-run output folder
        run_dir = out_dir / f'run_{run:04d}'
        run_dir.mkdir(exist_ok=True)

        # ground truth for comparison (take first batch element)
        gt_action = action[0].detach().cpu().numpy()

        for mode in modes:
            obs_dict = {'obs': obs, 'obs_img': obs_img, 'goal_img': goal_img}
            inf_time = None
            opt_time = 0.0

            if mode == 'disabled':
                obs_dict['adv_mask'] = torch.zeros(batch_size, device=device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    res = policy.predict_action(obs_dict)
                inf_time = time.perf_counter() - t0
                pred_action = res['action'][0].detach().cpu().numpy()

            elif mode == 'enabled':
                obs_dict['adv_mask'] = torch.ones(batch_size, device=device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    res = policy.predict_action(obs_dict)
                inf_time = time.perf_counter() - t0
                pred_action = res['action'][0].detach().cpu().numpy()

            else:  # optimized
                obs_dict['adv_mask'] = torch.ones(batch_size, device=device)
                t0 = time.perf_counter()
                # run optimization (may be slow)
                res_opt = policy.optimize_adversarial_intent(obs_dict, num_steps=args.opt_steps, lr=args.lr)
                opt_time = time.perf_counter() - t0
                pred_action = res_opt['action'][0].detach().cpu().numpy()
                inf_time = 0.0

            mse, gt_aligned, pred_aligned = align_and_mse(gt_action, pred_action)

            csv_writer.writerow([run, seed, mode, mse, gt_aligned.shape[0], pred_aligned.shape[0], f"{inf_time:.6f}", f"{opt_time:.6f}"])
            csv_file.flush()

            # save plot for this run+mode
            plot_path = run_dir / f'{mode}_traj.png'
            visualize_trajectory(gt_aligned, pred_aligned, plot_path, f'Run {run} - {mode} (MSE={mse:.6f})')

        print(f"Completed run {run+1}/{args.n_runs}")

    csv_file.close()

    # summary
    import pandas as pd
    df = pd.read_csv(csv_path)
    summary = df.groupby('mode')['mse'].agg(['mean', 'std', 'count']).reset_index()
    summary.to_csv(out_dir / 'summary.csv', index=False)
    print('Evaluation complete. Results saved to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-runs', type=int, default=10, help='Number of random runs')
    parser.add_argument('--output-dir', type=str, default='demo_eval', help='Directory to store results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--opt-steps', type=int, default=5, help='Optimization steps for adversarial intent')
    parser.add_argument('--lr', type=float, default=0.01, help='LR for adversarial optimization (if used)')
    args = parser.parse_args()
    main(args)
