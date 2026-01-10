import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wandb
from tqdm import tqdm
import sys
import copy
import matplotlib
matplotlib.use('Agg') # Set backend for headless environment
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage

# Add path to allow imports of vint_train package
# We need to add the directory CONTAINING vint_train, which is this script's directory
sys.path.append(os.path.dirname(__file__))
# Add 'diffusion_policy' root directory to path so we can import the inner diffusion_policy package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../diffusion_policy")))


# Import Project Modules
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.data.vint_dataset import ViNT_Dataset

# --- 1. 定义增强的数据集 Wrapper (Minimal Version for Ablation) ---
class ADSCD_DatasetWrapper(Dataset):
    """
    Wraps ViNT_Dataset. Simple pass-through for experimental scripts.
    """
    def __init__(self, vint_dataset, dataset_name, metadata_path=None, use_adv_data=True):
        self.dataset = vint_dataset
        self.use_adv_data = use_adv_data
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 1. Get Base Data
        # ViNT_Dataset returns (image, goal, dist, action) etc.
        data = self.dataset[idx] 
        obs_image = data[0]
        goal_image = data[1] 
        action = data[2]
        dist_label = data[3] 
        goal_pos = data[4]
        dataset_action_mask = data[6]
        
        # Mock Adv Inputs (ignored in this ablation)
        other_state = torch.zeros(2) 
        adv_mask = torch.tensor([0.0])
        
        return {
            'obs_image': obs_image,
            'goal_image': goal_image,
            'action': action,
            'dist_label': dist_label,
            'goal_pos': goal_pos,
            'dataset_action_mask': dataset_action_mask,
            'other_state': other_state,
            'adv_mask': adv_mask
        }

# --- 2. Ablation Model: Direct Features ---
class AblationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # A. Shared Vision Encoder (NoMaD-ViNT)
        print(f"Initializing Shared Vision Encoder: {config['model']['vision_encoder']}")
        self.vision_encoder = NoMaD_ViNT(
            obs_encoding_size=config['model']['obs_encoding_size'],
            context_size=config['model']['context_size'],
            mha_num_attention_heads=4, 
            mha_num_attention_layers=4,
            mha_ff_dim_factor=4
        )
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
            
        feature_dim = config['model']['obs_encoding_size'] # e.g. 256
        
        # B. Distance Head (Auxiliary)
        self.num_dist_classes = config['model'].get('num_dist_classes', 20)
        self.dist_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_dist_classes)
        )
        
        # C. Diffusion Decoder
        # DIRECTLY conditioned on visual features
        global_cond_dim = feature_dim
        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2, 
            global_cond_dim=global_cond_dim,
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
        """
        Forward pass for training.
        """
        # 1. Vision Features (256d)
        features = self.vision_encoder(obs_img, goal_img) 
        
        # 2. Distance Pred
        dist_pred = self.dist_head(features)
        
        return {
            'features': features,
            'dist_pred': dist_pred
        }

    @torch.no_grad()
    def get_action(self, obs_img, goal_img):
        device = obs_img.device
        B = obs_img.shape[0]
        
        # Get Features
        stats = self.forward(obs_img, goal_img)
        global_cond = stats['features']
        
        horizon = self.config['model']['len_traj_pred']
        noisy_action = torch.randn((B, horizon, 2), device=device)
        
        self.noise_scheduler.set_timesteps(self.config['model']['diffusion_iters'])
        for t in self.noise_scheduler.timesteps:
            noise_residual = self.noise_pred_net(
                noisy_action, 
                t, 
                global_cond=global_cond
            )
            noisy_action = self.noise_scheduler.step(noise_residual, t, noisy_action).prev_sample
            
        return noisy_action
    
    def compute_loss(self, batch, stage_cfg):
        obs_img = batch['obs_image']
        goal_img = batch['goal_image']
        action = batch['action']
        dist_label = batch.get('dist_label', None)
        dataset_action_mask = batch.get('dataset_action_mask', torch.ones(action.shape[0], device=action.device))
        
        # Encoder Forward
        stats = self.forward(obs_img, goal_img)
        global_cond = stats['features']
        dist_pred = stats['dist_pred']
        
        # Diffusion Training
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (action.shape[0],), device=action.device
        )
        
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        
        # Predict Noise
        noise_pred = self.noise_pred_net(
            noisy_action, 
            timesteps, 
            global_cond=global_cond
        )
        
        # 1. Diffusion Loss
        diff_loss_full = nn.functional.mse_loss(noise_pred, noise, reduction='none')
        diff_loss_sample = diff_loss_full.mean(dim=(1, 2)) 
        
        mask = dataset_action_mask.view(-1)
        diff_loss = (diff_loss_sample * mask).sum() / (mask.sum() + 1e-6)
        
        # 2. Dist Loss
        dist_loss = torch.tensor(0.0, device=action.device)
        if dist_label is not None:
             ce_loss = nn.CrossEntropyLoss()
             dist_label_clipped = torch.clamp(dist_label, min=0, max=self.num_dist_classes-1)
             dist_loss = ce_loss(dist_pred, dist_label_clipped)

        total_loss = diff_loss + 0.1 * dist_loss 
                     
        return {
            'loss': total_loss,
            'diff_loss': diff_loss,
            'dist_loss': dist_loss
        }

# --- Visualization Utils ---
def visualize_trajectory(gt_action, pred_action, goal_pos=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([0], [0], c='black', marker='o', label='Robot Start', s=100)
    ax.plot(gt_action[:, 0], gt_action[:, 1], 'g.-', label='Ground Truth', linewidth=2, markersize=10)
    if goal_pos is not None:
        ax.scatter(goal_pos[0], goal_pos[1], c='blue', marker='x', s=200, label='Subgoal', linewidths=3)
    else:
        ax.scatter(gt_action[-1, 0], gt_action[-1, 1], c='green', marker='*', s=150, label='GT End')
    ax.plot(pred_action[:, 0], pred_action[:, 1], 'r.-', label='Prediction', linewidth=2, markersize=10)
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PILImage.open(buf)
    plt.close(fig)
    return wandb.Image(image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(f"cuda:{cfg['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading datasets (Mock-Safe)...")
    try:
        data_config = cfg['dataset']
        base_dataset = ViNT_Dataset(
            data_folder=data_config.get('data_folder', '.'),
            data_split_folder=data_config.get('split_path', '.'),
            dataset_name=data_config.get('name', 'gazebo'),
            image_size=cfg['model']['image_size'],
            waypoint_spacing=1,
            min_dist_cat=0, max_dist_cat=5,
            min_action_distance=1, max_action_distance=5,
            negative_mining=True, len_traj_pred=cfg['model']['len_traj_pred'],
            learn_angle=False, context_size=cfg['model']['context_size'],
            context_type="temporal", goals_per_obs=1, normalize=True,
            obs_type="image", goal_type="image"
        )
    except Exception as e:
        print(f"Warning: Could not load real ViNT Dataset ({e}).")
        return

    dataset = ADSCD_DatasetWrapper(base_dataset, dataset_name=data_config['name'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    
    # Init Ablation Model
    model = AblationModel(cfg).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']))
    
    if cfg['use_wandb']:
        wandb.init(project=cfg['project_name'], name=f"{cfg['run_name']}_ablation_direct", config=cfg)
        
    global_step = 0
    model.train()
    
    # Run shorter/same training
    for epoch in range(10): 
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            loss_dict = model.compute_loss(batch, None) # stage_cfg not needed for ablation
            loss = loss_dict['loss']
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item()})
            
            if cfg['use_wandb']:
                log_data = loss_dict.copy()
                if global_step % 500 == 0:
                    model.eval()
                    with torch.no_grad():
                        viz_gt_action = batch['action'][0].cpu().numpy() 
                        viz_pred_action = model.get_action(batch['obs_image'][0:1], batch['goal_image'][0:1])[0].cpu().numpy()
                        log_data["viz/trajectory"] = visualize_trajectory(viz_gt_action, viz_pred_action, None)
                    model.train()
                wandb.log(log_data, step=global_step)
        
        # Save
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint_ablation_direct_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()
