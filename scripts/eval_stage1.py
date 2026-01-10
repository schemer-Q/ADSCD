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

# Set non-interactive backend
matplotlib.use('Agg')

# Add 'train' directory to path to import vint_train modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../train")))
# Add 'diffusion_policy' root directory to path so we can import the inner diffusion_policy package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../diffusion_policy")))

from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.data.vint_dataset import ViNT_Dataset
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

# --- Classes Copied from train_adscd.py to ensure consistency ---

class ADSCD_DatasetWrapper(Dataset):
    """
    Wraps ViNT_Dataset to add 'other_state' and 'adv_mask'.
    Supports loading from external metadata files.
    """
    def __init__(self, vint_dataset, dataset_name, metadata_path=None, use_adv_data=True):
        self.dataset = vint_dataset
        self.use_adv_data = use_adv_data
        self.metadata = None

        # Load Sidecar Metadata if provided
        if self.use_adv_data and metadata_path and os.path.exists(metadata_path):
            print(f"[{dataset_name}] Loading Adv Metadata from {metadata_path}...")
            import pickle
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 1. Get Base Data
        data = self.dataset[idx] 
        obs_image = data[0]
        goal_image = data[1]
        action = data[2]
        goal_pos = data[4]
        dataset_action_mask = data[6]
        
        # 2. Add 'Other State' and 'Adv Mask'
        other_state = torch.zeros(2) 
        adv_mask = torch.tensor([0.0])
        
        if self.use_adv_data:
            if self.metadata is not None:
                if hasattr(self.dataset, 'index_to_data'):
                    traj_name, curr_time, _ = self.dataset.index_to_data[idx]
                    key = (traj_name, curr_time)
                    if key in self.metadata:
                        meta = self.metadata[key]
                        adv_mask = torch.tensor([meta['adv_mask']])
                        other_state = torch.tensor(meta['other_state']).float()
            else:
                if np.random.rand() > 0.8:
                    adv_mask = torch.tensor([1.0])
                    other_state = torch.randn(2)
        
        return {
            'obs_image': obs_image,
            'goal_image': goal_image,
            'action': action,
            'goal_pos': goal_pos,
            'dataset_action_mask': dataset_action_mask,
            'other_state': other_state,
            'adv_mask': adv_mask
        }

class ADSCDModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # A. Shared Vision Encoder (ViT / NoMaD-ViNT)
        print(f"Initializing Shared Vision Encoder: {config['model']['vision_encoder']}")
        if config['model']['vision_encoder'] == 'nomad_vint':
            self.vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config['model']['obs_encoding_size'],
                context_size=config['model']['context_size'],
                mha_num_attention_heads=4, 
                mha_num_attention_layers=4,
                mha_ff_dim_factor=4
            )
            self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
        else:
            raise NotImplementedError
            
        feature_dim = config['model']['obs_encoding_size']
        
        # B. Nav Head
        self.nav_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config['model']['z_nav_dim'] * 2) 
        )
        
        # C. Adv Head
        other_state_dim = 2 
        self.adv_head = nn.Sequential(
            nn.Linear(feature_dim + other_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config['model']['z_adv_dim'] * 2)
        )
        
        # D. Diffusion Decoder
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
        
        return {
            'z_cond': z_cond,
            'nav_dist': (nav_mu, nav_logvar),
            'adv_dist_real': (adv_mu, adv_logvar)
        }

    @torch.no_grad()
    def get_action(self, obs_img, goal_img, other_state=None, adv_mask=None):
        device = obs_img.device
        B = obs_img.shape[0]
        
        if adv_mask is None:
            adv_mask = torch.zeros((B, 1), device=device)
        if other_state is None:
            other_state = torch.zeros((B, 2), device=device)
            
        stats = self.forward(obs_img, goal_img, other_state, adv_mask)
        z_cond = stats['z_cond']
        
        horizon = self.config['model']['len_traj_pred']
        noisy_action = torch.randn((B, horizon, 2), device=device)
        
        self.noise_scheduler.set_timesteps(self.config['model']['diffusion_iters'])
        for t in self.noise_scheduler.timesteps:
            noise_residual = self.noise_pred_net(noisy_action, t, global_cond=z_cond)
            noisy_action = self.noise_scheduler.step(noise_residual, t, noisy_action).prev_sample
            
        return noisy_action

# --- Evaluation Logic ---

def compute_metrics(pred_traj, gt_traj, mask):
    """
    pred_traj: (B, H, 2)
    gt_traj: (B, H, 2)
    mask: (B,) or (B, 1) or (B, H) depending on dataset. 
          Assuming (B,) indicating valid sample based on training code.
    """
    displacement_error = torch.norm(pred_traj - gt_traj, dim=2) # (B, H)
    ade_per_sample = displacement_error.mean(dim=1) # (B,)
    fde_per_sample = displacement_error[:, -1] # (B,)
    
    mask = mask.view(-1)
    
    valid_ade = ade_per_sample * mask
    valid_fde = fde_per_sample * mask
    
    return valid_ade.sum().item(), valid_fde.sum().item(), mask.sum().item()

def visualize_results(obs_img, goal_img, gt_traj, pred_traj, save_path):
    """
    Save specific examples
    """
    # Unnormalize images for display
    # ViNT dataset images are roughly [-1, 1] or normalized.
    # Assuming standard imagenet normalization or similar.
    # For vis, just clip.
    
    B = obs_img.shape[0]
    
    fig, axes = plt.subplots(B, 3, figsize=(15, 5*B))
    if B == 1:
        axes = axes[np.newaxis, :]
        
    for i in range(B):
        # 1. Obs
        img = obs_img[i].cpu().permute(1, 2, 0).numpy()
        # Handle Context Stacking (take last 3 channels)
        if img.shape[2] > 3:
            img = img[:, :, -3:]
            
        # Denormalize approximation
        img = (img * 0.229 + 0.485) # ImageNet un-norm
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Observation")
        
        # 2. Goal
        g_img = goal_img[i].cpu().permute(1, 2, 0).numpy()
        # Handle Context Stacking (take last 3 channels, though goal is usually single image)
        if g_img.shape[2] > 3:
            g_img = g_img[:, :, -3:]
            
        g_img = (g_img * 0.229 + 0.485)
        g_img = np.clip(g_img, 0, 1)
        axes[i, 1].imshow(g_img)
        axes[i, 1].set_title("Goal")
        
        # 3. Trajectory
        # Create blank plot or overlay? Overlay is hard without transform. Just plot scatter.
        axes[i, 2].plot(gt_traj[i, :, 0].cpu(), gt_traj[i, :, 1].cpu(), 'g.-', label='Ground Truth')
        axes[i, 2].plot(pred_traj[i, :, 0].cpu(), pred_traj[i, :, 1].cpu(), 'r.-', label='Predicted')
        axes[i, 2].legend()
        axes[i, 2].set_title(f"Trajectory (Blue=GT, Red=Pred)")
        axes[i, 2].set_aspect('equal')
        axes[i, 2].grid(True)
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

def eval_epoch(model, loader, device, max_batches=None, viz_batch=False, output_dir="eval_output"):
    model.eval()
    
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0.0
    all_nav_mus = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
                
            obs_img = batch['obs_image'].to(device)
            goal_img = batch['goal_image'].to(device)
            action = batch['action'].to(device)
            adv_mask = batch['adv_mask'].to(device)
            other_state = batch['other_state'].to(device)
            dataset_action_mask = batch.get('dataset_action_mask', torch.ones(action.shape[0])).to(device)
            
            # Predict
            # We need stats here too
            stats = model.forward(obs_img, goal_img, other_state, adv_mask)
            z_cond = stats['z_cond']
            nav_mu, _ = stats['nav_dist']
            all_nav_mus.append(nav_mu.cpu())
            
            # Diffusion Inference
            horizon = model.config['model']['len_traj_pred']
            noisy_action = torch.randn((obs_img.shape[0], horizon, 2), device=device)
            model.noise_scheduler.set_timesteps(model.config['model']['diffusion_iters'])
            
            for t in model.noise_scheduler.timesteps:
                noise_residual = model.noise_pred_net(noisy_action, t, global_cond=z_cond)
                noisy_action = model.noise_scheduler.step(noise_residual, t, noisy_action).prev_sample
            
            pred_action = noisy_action
            
            # Compute Metrics
            sum_ade, sum_fde, n_valid = compute_metrics(pred_action, action, dataset_action_mask)
            total_ade += sum_ade
            total_fde += sum_fde
            total_samples += n_valid
            
            # Visualize first batch (only valid samples preferably, but visualization logic can stay simple)
            if viz_batch and batch_idx == 0:
                visualize_results(
                    obs_img[:4], 
                    goal_img[:4], 
                    action[:4], 
                    pred_action[:4], 
                    os.path.join(output_dir, "eval_vis.png")
                )
                
    avg_ade = total_ade / (total_samples + 1e-6)
    avg_fde = total_fde / (total_samples + 1e-6)
    
    return avg_ade, avg_fde, all_nav_mus

def plot_latent_distribution(nav_mus, save_path):
    """
    nav_mus: list of tensors or big tensor (N, z_dim)
    """
    if isinstance(nav_mus, list):
        nav_mus = torch.cat(nav_mus, dim=0)
    
    nav_mus = nav_mus.cpu().numpy()
    
    # Plot histogram of first few dimensions
    dim = min(4, nav_mus.shape[1])
    fig, axes = plt.subplots(1, dim, figsize=(5*dim, 4))
    if dim == 1:
        axes = [axes]
        
    for i in range(dim):
        axes[i].hist(nav_mus[:, i], bins=50, alpha=0.7, color='blue')
        axes[i].set_title(f"Nav Z dim {i}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved latent plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config .yaml')
    parser.add_argument('--split_dir', type=str, default=None, help='Override split dir')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='eval_outputs')
    
    args = parser.parse_args()
    
    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(args.device)
    
    # Init Model
    model = ADSCDModel(config)
    model.to(device)
    
    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle state_dict key mismatch if saved as 'model_state_dict' or just raw
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    print("Model loaded successfully.")
    
    # Init Dataset (TEST split)
    data_folder = config['dataset']['data_folder']
    
    # Determine split path (Train config points to .../split/train, we want .../split/test)
    if args.split_dir:
        test_split_path = args.split_dir
    else:
        # Infer from config: ../train -> ../test
        train_split = config['dataset']['split_path']
        parent = os.path.dirname(train_split)
        test_split_path = os.path.join(parent, "test")
        
    print(f"Loading Test Dataset from {data_folder} using split {test_split_path}")
    
    test_dataset = ViNT_Dataset(
        data_folder=data_folder,
        data_split_folder=test_split_path,
        dataset_name=config['dataset']['name'],
        image_size=config['model']['image_size'],
        waypoint_spacing=1,
        min_dist_cat=0, 
        max_dist_cat=20, # Use larger range for test if desired, or match train (0-5)
        min_action_distance=1, 
        max_action_distance=20, # Allow longer trajectories
        negative_mining=False, # Consistently valid goals for eval
        len_traj_pred=config['model']['len_traj_pred'],
        learn_angle=False, 
        context_size=config['model']['context_size'],
        context_type="temporal", 
        goals_per_obs=1, 
        normalize=True,
        obs_type="image", 
        goal_type="image"
    )
    
    # Wrap
    test_loader_dataset = ADSCD_DatasetWrapper(
        test_dataset, 
        config['dataset']['name'], 
        use_adv_data=False # Stage 1 evaluation typically pure nav, but wrapper handles structure
    )
    
    test_loader = DataLoader(
        test_loader_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"Test Dataset Size: {len(test_dataset)}")
    
    # Run Evaluation
    ade, fde, nav_mus = eval_epoch(model, test_loader, device, viz_batch=True, output_dir=args.output_dir)
    
    # Plot Latents
    plot_latent_distribution(nav_mus, os.path.join(args.output_dir, "latent_dist.png"))
    
    print("="*30)
    print(f"Evaluation Results for {args.checkpoint}")
    print(f"ADE: {ade:.4f}")
    print(f"FDE: {fde:.4f}")
    print(f"Visualizations saved to {args.output_dir}")
    print("="*30)

if __name__ == '__main__':
    main()
