import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib
import io
from PIL import Image as PILImage

# Set non-interactive backend
matplotlib.use('Agg')

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../train")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../diffusion_policy")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.data.vint_dataset import ViNT_Dataset
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

# --- Dataset Wrapper ---
class ADSCD_DatasetWrapper(Dataset):
    def __init__(self, vint_dataset, dataset_name, metadata_path=None, use_adv_data=True):
        self.dataset = vint_dataset
        self.use_adv_data = use_adv_data
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx] 
        # Adapt to index changes if any, assuming standard ViNT
        obs_image = data[0]
        goal_image = data[1]
        action = data[2]
        # data[3] is dist, data[4] is goal_pos
        dataset_action_mask = data[6] # Usually
        
        return {
            'obs_image': obs_image,
            'goal_image': goal_image,
            'action': action,
            'dataset_action_mask': dataset_action_mask,
            'other_state': torch.zeros(2),
            'adv_mask': torch.tensor([0.0])
        }

# --- Baseline Model (ADSCDModel) ---
class ADSCDModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # A. Shared Vision Encoder
        if config['model']['vision_encoder'] == 'nomad_vint':
            self.vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config['model']['obs_encoding_size'],
                context_size=config['model']['context_size'],
                mha_num_attention_heads=4, 
                mha_num_attention_layers=4,
                mha_ff_dim_factor=4
            )
            self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
            
        feature_dim = config['model']['obs_encoding_size']
        
        # B. Nav Head
        self.nav_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config['model']['z_nav_dim'] * 2) 
        )
        
        # B.2 Distance Head (Added in last iteration)
        self.num_dist_classes = config['model'].get('num_dist_classes', 20)
        self.dist_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_dist_classes)
        )
        
        # C. Adv Head
        other_state_dim = 2 
        self.adv_head = nn.Sequential(
            nn.Linear(feature_dim + other_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config['model']['z_adv_dim'] * 2) 
        )
        
        # D. Diffusion
        z_total_dim = config['model']['z_nav_dim'] + config['model']['z_adv_dim']
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2, 
            global_cond_dim=z_total_dim,
            down_dims=config['model']['down_dims'],
            cond_predict_scale=False
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config['model']['diffusion_iters'],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs_img, goal_img, other_state, adv_mask):
        features = self.vision_encoder(obs_img, goal_img) 
        nav_mu_logvar = self.nav_head(features)
        nav_mu, nav_logvar = torch.chunk(nav_mu_logvar, 2, dim=1)
        z_nav = self.reparameterize(nav_mu, nav_logvar)
        
        other_state_clean = other_state * adv_mask 
        adv_input = torch.cat([features, other_state_clean], dim=1)
        adv_mu_logvar = self.adv_head(adv_input)
        adv_mu, adv_logvar = torch.chunk(adv_mu_logvar, 2, dim=1)
        z_adv = self.reparameterize(adv_mu, adv_logvar)
        
        z_adv_masked = z_adv * adv_mask
        z_cond = torch.cat([z_nav, z_adv_masked], dim=1)
        
        return {'z_cond': z_cond}

    @torch.no_grad()
    def get_action(self, obs_img, goal_img, other_state=None, adv_mask=None):
        device = obs_img.device
        B = obs_img.shape[0]
        if adv_mask is None: adv_mask = torch.zeros((B, 1), device=device)
        if other_state is None: other_state = torch.zeros((B, 2), device=device)
            
        stats = self.forward(obs_img, goal_img, other_state, adv_mask)
        z_cond = stats['z_cond']
        
        horizon = self.config['model']['len_traj_pred']
        noisy_action = torch.randn((B, horizon, 2), device=device)
        self.noise_scheduler.set_timesteps(self.config['model']['diffusion_iters'])
        for t in self.noise_scheduler.timesteps:
            noise_residual = self.noise_pred_net(noisy_action, t, global_cond=z_cond)
            noisy_action = self.noise_scheduler.step(noise_residual, t, noisy_action).prev_sample
        return noisy_action

# --- Ablation Model ---
class AblationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config['model']['vision_encoder'] == 'nomad_vint':
            self.vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config['model']['obs_encoding_size'],
                context_size=config['model']['context_size'],
                mha_num_attention_heads=4, 
                mha_num_attention_layers=4,
                mha_ff_dim_factor=4
            )
            self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
            
        feature_dim = config['model']['obs_encoding_size']
        
        # B. Distance Head (Auxiliary)
        self.num_dist_classes = config['model'].get('num_dist_classes', 20)
        self.dist_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_dist_classes)
        )
        
        # Diffusion directly on features
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2, 
            global_cond_dim=feature_dim, # 256
            down_dims=config['model']['down_dims'],
            cond_predict_scale=False 
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config['model']['diffusion_iters'],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def forward(self, obs_img, goal_img):
        features = self.vision_encoder(obs_img, goal_img) 
        return {'features': features}

    @torch.no_grad()
    def get_action(self, obs_img, goal_img):
        device = obs_img.device
        B = obs_img.shape[0]
        stats = self.forward(obs_img, goal_img)
        global_cond = stats['features']
        
        horizon = self.config['model']['len_traj_pred']
        noisy_action = torch.randn((B, horizon, 2), device=device)
        self.noise_scheduler.set_timesteps(self.config['model']['diffusion_iters'])
        for t in self.noise_scheduler.timesteps:
            noise_residual = self.noise_pred_net(noisy_action, t, global_cond=global_cond)
            noisy_action = self.noise_scheduler.step(noise_residual, t, noisy_action).prev_sample
        return noisy_action

# --- Metrics ---
def compute_metrics(pred_traj, gt_traj, mask):
    displacement_error = torch.norm(pred_traj - gt_traj, dim=2) # (B, H)
    ade_per_sample = displacement_error.mean(dim=1) # (B,)
    fde_per_sample = displacement_error[:, -1] # (B,)
    
    mask = mask.view(-1)
    valid_ade = ade_per_sample * mask
    valid_fde = fde_per_sample * mask
    
    return valid_ade.sum().item(), valid_fde.sum().item(), mask.sum().item()

# --- Viz ---
def visualize_comparison(obs_img, goal_img, gt_traj, pred_base, pred_ablation, save_path):
    B = obs_img.shape[0]
    # Limit to 5 samples max
    B = min(B, 5)
    
    fig, axes = plt.subplots(B, 3, figsize=(15, 5*B))
    if B == 1: axes = axes[np.newaxis, :]
        
    for i in range(B):
        # 1. Obs
        img = obs_img[i].cpu().permute(1, 2, 0).numpy()
        if img.shape[2] > 3: img = img[:, :, -3:]
        img = (img * 0.229 + 0.485)
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Observation")
        
        # 2. Goal
        g_img = goal_img[i].cpu().permute(1, 2, 0).numpy()
        if g_img.shape[2] > 3: g_img = g_img[:, :, -3:]
        g_img = (g_img * 0.229 + 0.485)
        g_img = np.clip(g_img, 0, 1)
        axes[i, 1].imshow(g_img)
        axes[i, 1].set_title("Goal")
        
        # 3. Trajectory Comparison
        axes[i, 2].plot(gt_traj[i, :, 0].cpu(), gt_traj[i, :, 1].cpu(), 'g.-', label='GT', linewidth=2)
        axes[i, 2].plot(pred_base[i, :, 0].cpu(), pred_base[i, :, 1].cpu(), 'b.-', label='Baseline (z_nav)', linewidth=2)
        axes[i, 2].plot(pred_ablation[i, :, 0].cpu(), pred_ablation[i, :, 1].cpu(), 'r.-', label='Ablation (Direct)', linewidth=2)
        
        # Mark Endpoints
        axes[i, 2].scatter(gt_traj[i, -1, 0].cpu(), gt_traj[i, -1, 1].cpu(), c='g', marker='*')
        axes[i, 2].scatter(pred_base[i, -1, 0].cpu(), pred_base[i, -1, 1].cpu(), c='b', marker='o')
        axes[i, 2].scatter(pred_ablation[i, -1, 0].cpu(), pred_ablation[i, -1, 1].cpu(), c='r', marker='x')

        if i == 0: axes[i, 2].legend()
        axes[i, 2].set_title(f"Trajectory Comparison")
        axes[i, 2].set_aspect('equal')
        axes[i, 2].grid(True)
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt_base', type=str, required=True)
    parser.add_argument('--ckpt_ablation', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='eval_compare_output')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Load Models
    print("Loading Baseline Model...")
    model_base = ADSCDModel(config).to(device)
    model_base.load_state_dict(torch.load(args.ckpt_base, map_location=device))
    model_base.eval()
    
    print("Loading Ablation Model...")
    model_ablation = AblationModel(config).to(device)
    model_ablation.load_state_dict(torch.load(args.ckpt_ablation, map_location=device))
    model_ablation.eval()

    # 2. Load Data
    data_folder = config['dataset']['data_folder']
    train_split = config['dataset']['split_path']
    test_split_path = os.path.join(os.path.dirname(train_split), "test")
    
    print(f"Loading Test Data from {test_split_path}")
    test_dataset = ViNT_Dataset(
        data_folder=data_folder,
        data_split_folder=test_split_path,
        dataset_name=config['dataset']['name'],
        image_size=config['model']['image_size'],
        waypoint_spacing=1,
        min_dist_cat=0, max_dist_cat=20,
        min_action_distance=1, max_action_distance=20,
        negative_mining=False,
        len_traj_pred=config['model']['len_traj_pred'],
        learn_angle=False, 
        context_size=config['model']['context_size'],
        context_type="temporal", 
        goals_per_obs=1, 
        normalize=True,
        obs_type="image", 
        goal_type="image"
    )
    test_loader = DataLoader(
        ADSCD_DatasetWrapper(test_dataset, "test", use_adv_data=False),
        batch_size=32, shuffle=False, num_workers=4
    )

    # 3. Eval Loop
    metrics_base = {'ade': 0.0, 'fde': 0.0, 'count': 0}
    metrics_ablation = {'ade': 0.0, 'fde': 0.0, 'count': 0}
    
    viz_done = False
    
    print("Starting Comparison...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            obs = batch['obs_image'].to(device)
            goal = batch['goal_image'].to(device)
            action = batch['action'].to(device)
            mask = batch['dataset_action_mask'].to(device)
            
            # Baseline Pred
            pred_base = model_base.get_action(obs, goal, None, None)
            
            # Ablation Pred
            pred_ablation = model_ablation.get_action(obs, goal)
            
            # Metrics
            ade_b, fde_b, n_b = compute_metrics(pred_base, action, mask)
            metrics_base['ade'] += ade_b
            metrics_base['fde'] += fde_b
            metrics_base['count'] += n_b
            
            ade_a, fde_a, n_a = compute_metrics(pred_ablation, action, mask)
            metrics_ablation['ade'] += ade_a
            metrics_ablation['fde'] += fde_a
            metrics_ablation['count'] += n_a
            
            # Viz first valid batch
            if not viz_done and n_b > 0:
                visualize_comparison(
                    obs, goal, action, pred_base, pred_ablation, 
                    os.path.join(args.output_dir, "comparison_viz.png")
                )
                viz_done = True
                
    # Report
    print("\n" + "="*40)
    print("COMPARISON RESULTS (Test Set)")
    print("="*40)
    
    print(f"BASELINE MODEL (ADSCD w/ z_nav):")
    print(f"  ADE: {metrics_base['ade']/metrics_base['count']:.4f}")
    print(f"  FDE: {metrics_base['fde']/metrics_base['count']:.4f}")
    
    print("-" * 40)
    
    print(f"ABLATION MODEL (Direct Features):")
    print(f"  ADE: {metrics_ablation['ade']/metrics_ablation['count']:.4f}")
    print(f"  FDE: {metrics_ablation['fde']/metrics_ablation['count']:.4f}")
    
    print("="*40)
    print(f"Visualization saved to {args.output_dir}/comparison_viz.png")

if __name__ == "__main__":
    main()
