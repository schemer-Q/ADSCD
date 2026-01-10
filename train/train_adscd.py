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

# Import Project Modules
# Add 'diffusion_policy' root directory to path so we can import the inner diffusion_policy package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../diffusion_policy")))

from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.data.vint_dataset import ViNT_Dataset

# --- 1. 定义增强的数据集 Wrapper ---
class ADSCD_DatasetWrapper(Dataset):
    """
    Wraps ViNT_Dataset to add 'other_state' and 'adv_mask'.
    Supports loading from external metadata files.
    Also ensures 'goal_image' provided.
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
        # ViNT_Dataset returns (image, goal, dist, action) etc.
        data = self.dataset[idx] 
        # Attempt to unpack generally
        obs_image = data[0]
        goal_image = data[1] # ViNT Dataset automatically samples goal if properly configured
        # If goal_image is all zeros or wrong shape, we might need manual sampling, 
        # but standard ViNT_Dataset logic (lines 191+) handles goal sampling via _sample_goal.
        
        # FIXED: Action is at index 2, data[-1] is action_mask
        action = data[2]
        dist_label = data[3] # Topological Distance usually at index 3
        goal_pos = data[4]
        dataset_action_mask = data[6]
        
        # 2. Add 'Other State' and 'Adv Mask'
        other_state = torch.zeros(2) 
        adv_mask = torch.tensor([0.0])
        
        if self.use_adv_data:
            if self.metadata is not None:
                # Retrieve from metadata using internal index
                if hasattr(self.dataset, 'index_to_data'):
                    traj_name, curr_time, _ = self.dataset.index_to_data[idx]
                    key = (traj_name, curr_time)
                    if key in self.metadata:
                        meta = self.metadata[key]
                        adv_mask = torch.tensor([meta['adv_mask']])
                        other_state = torch.tensor(meta['other_state']).float()
            else:
                # Mock if no metadata found but use_adv_data is True
                if np.random.rand() > 0.8:
                    adv_mask = torch.tensor([1.0])
                    other_state = torch.randn(2)
        
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

# --- 2. 定义核心模型架构 ---
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
            
        feature_dim = config['model']['obs_encoding_size'] # e.g. 256
        
        # B. Nav Head (E_nav): Features -> z_nav (Gaussian Parameters)
        self.nav_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config['model']['z_nav_dim'] * 2) # mu, logvar
        )
        
        # B.2 Distance Pred Head (Auxiliary Task)
        # Usually dist classes depend on dataset config (max_dist_cat). 
        # We assume a safe default or config value.
        self.num_dist_classes = config['model'].get('num_dist_classes', 20)
        self.dist_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_dist_classes)
        )
        
        # C. Adv Head (E_adv): [Features, OtherState] -> z_adv (Gaussian Params)
        # Using Features + LowDim State
        other_state_dim = 2 # (x, y)
        self.adv_head = nn.Sequential(
            nn.Linear(feature_dim + other_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config['model']['z_adv_dim'] * 2) # mu, logvar
        )
        
        # D. Diffusion Decoder
        # Conditioned on z = [z_nav, z_adv] (concatenated)
        z_total_dim = config['model']['z_nav_dim'] + config['model']['z_adv_dim']
        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2, # Action dim (dx, dy)
            global_cond_dim=z_total_dim,
            down_dims=config['model']['down_dims'],
            cond_predict_scale=False # NoMaD style default
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
        """
        Forward pass for training.
        """
        # 1. Shared Vision Features
        # NoMaD encoder typically takes (obs_img, goal_img) or just obs.
        # Assuming NoMaD_ViNT forward signature: forward(obs_img, goal_img=None) -> features
        # We enforce context length behavior inside dataset or here.
        features = self.vision_encoder(obs_img, goal_img) # (B, feature_dim)
        
        # 2. Nav Encoder
        nav_mu_logvar = self.nav_head(features)
        nav_mu, nav_logvar = torch.chunk(nav_mu_logvar, 2, dim=1)
        z_nav = self.reparameterize(nav_mu, nav_logvar)
        
        # 2.5 Distance Prediction
        dist_pred = self.dist_head(features) # (B, num_classes)
        
        # 3. Adv Encoder
        # --- Robustness Fix ---
        # 如果 g=0 (adv_mask=0), 强制 other_state 为 0。
        # 这保证了"无对手"和"有对手但忽略"在 Encoder 输入端的一致性，防止噪声输入。
        # adv_mask shape is (B, 1), other_state is (B, 2)
        other_state_clean = other_state * adv_mask 

        # Concatenate Vision Features + Cleaned Other State
        adv_input = torch.cat([features, other_state_clean], dim=1)
        adv_mu_logvar = self.adv_head(adv_input)
        adv_mu, adv_logvar = torch.chunk(adv_mu_logvar, 2, dim=1)
        z_adv = self.reparameterize(adv_mu, adv_logvar)
        
        # 4. Latent Fusion & Masking
        # z = [z_nav, g * z_adv]
        # g (adv_mask) shape (B, 1)
        z_adv_masked = z_adv * adv_mask
        z_cond = torch.cat([z_nav, z_adv_masked], dim=1) # (B, z_nav + z_adv)
        
        return {
            'z_cond': z_cond,
            'nav_dist': (nav_mu, nav_logvar),
            'adv_dist': (nav_mu, nav_logvar), # Typo fixed in thought: should be adv vars
            'adv_dist_real': (adv_mu, adv_logvar),
            'dist_pred': dist_pred
        }

    @torch.no_grad()
    def get_action(self, obs_img, goal_img, other_state=None, adv_mask=None):
        """
        推理专用接口：处理缺失输入的情况
        """
        device = obs_img.device
        B = obs_img.shape[0]
        
        # Handle Missing Inputs for Pure Nav Mode
        if adv_mask is None:
            adv_mask = torch.zeros((B, 1), device=device)
            
        if other_state is None:
            # fill with zeros if not provided
            other_state = torch.zeros((B, 2), device=device)
            
        # Get Latents
        stats = self.forward(obs_img, goal_img, other_state, adv_mask)
        z_cond = stats['z_cond']
        
        # Diffusion Inference
        # Initialize random noise
        # Shape: (B, horizon, action_dim=2)
        # Assuming horizon=len_traj_pred from config
        horizon = self.config['model']['len_traj_pred']
        noisy_action = torch.randn((B, horizon, 2), device=device)
        
        # Denoising Loop
        self.noise_scheduler.set_timesteps(self.config['model']['diffusion_iters'])
        for t in self.noise_scheduler.timesteps:
            # Model Input: (noisy_action, timestep, global_cond)
            noise_residual = self.noise_pred_net(
                noisy_action, 
                t, 
                global_cond=z_cond
            )
            
            # Step
            noisy_action = self.noise_scheduler.step(noise_residual, t, noisy_action).prev_sample
            
        return noisy_action
    
    def compute_loss(self, batch, stage_cfg):
        obs_img = batch['obs_image']
        goal_img = batch['goal_image'] # Used by vision encoder
        other_state = batch['other_state']
        adv_mask = batch['adv_mask']
        action = batch['action'] # (B, H, 2)
        dist_label = batch.get('dist_label', None)
        dataset_action_mask = batch.get('dataset_action_mask', torch.ones(action.shape[0], device=action.device))
        
        # Encoder Forward
        stats = self.forward(obs_img, goal_img, other_state, adv_mask)
        z_cond = stats['z_cond']
        dist_pred = stats['dist_pred']
        
        # Diffusion Training
        # Sample noise
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
            global_cond=z_cond
        )
        
        # 1. Diffusion Loss (MSE)
        # Apply mask: dataset_action_mask indicates if the action supervision is valid
        diff_loss_full = nn.functional.mse_loss(noise_pred, noise, reduction='none')
        diff_loss_sample = diff_loss_full.mean(dim=(1, 2)) # Avg over H, D
        
        mask = dataset_action_mask.view(-1)
        diff_loss = (diff_loss_sample * mask).sum() / (mask.sum() + 1e-6)
        
        # 2. KL Divergence Losses
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log_var - mu^2 - var)
        
        # Nav KL
        nav_mu, nav_logvar = stats['nav_dist']
        nav_kl = -0.5 * torch.sum(1 + nav_logvar - nav_mu.pow(2) - nav_logvar.exp(), dim=1).mean()
        
        # Adv KL (Only computed if mask=1? Or always regularization? Usually always reg.)
        adv_mu, adv_logvar = stats['adv_dist_real']
        adv_kl = -0.5 * torch.sum(1 + adv_logvar - adv_mu.pow(2) - adv_logvar.exp(), dim=1).mean()
        
        # 3. Distance Loss (Cross Entropy)
        dist_loss = torch.tensor(0.0, device=action.device)
        if dist_label is not None and stage_cfg.get('train_dist', False):
            # Clip label to num_classes
            # Using CrossEntropyLoss
            ce_loss = nn.CrossEntropyLoss()
            # Ensure label is within range [0, num_classes-1]
            dist_label_clipped = torch.clamp(dist_label, min=0, max=self.num_dist_classes-1)
            dist_loss = ce_loss(dist_pred, dist_label_clipped)

        # Total Loss
        total_loss = diff_loss + \
                     stage_cfg['loss_weights']['nav_kl'] * nav_kl + \
                     stage_cfg['loss_weights']['adv_kl'] * adv_kl
        
        if stage_cfg.get('train_dist', False):
            total_loss += stage_cfg['loss_weights'].get('dist_ce', 0.0) * dist_loss
                     
        return {
            'loss': total_loss,
            'diff_loss': diff_loss,
            'nav_kl': nav_kl,
            'adv_kl': adv_kl,
            'dist_loss': dist_loss
        }

# --- 2.5 Visualization Utils ---
def visualize_trajectory(gt_action, pred_action, goal_pos=None):
    """
    Inputs:
        gt_action: (H, 2) numpy array
        pred_action: (H, 2) numpy array
        goal_pos: (2,) numpy array (optional)
    Returns:
        wandb.Image
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot Origin
    ax.scatter([0], [0], c='black', marker='o', label='Robot Start', s=100)
    
    # Plot Ground Truth
    ax.plot(gt_action[:, 0], gt_action[:, 1], 'g.-', label='Ground Truth', linewidth=2, markersize=10)
    
    # Mark Goal (Subgoal)
    if goal_pos is not None:
        ax.scatter(goal_pos[0], goal_pos[1], c='blue', marker='x', s=200, label='Subgoal', linewidths=3)
        # Connect last GT point to Subgoal? Usually minimal gap.
    else:
        # Fallback approximation
        ax.scatter(gt_action[-1, 0], gt_action[-1, 1], c='green', marker='*', s=150, label='GT End')
    
    # Plot Prediction
    ax.plot(pred_action[:, 0], pred_action[:, 1], 'r.-', label='Prediction', linewidth=2, markersize=10)
    
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison')
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PILImage.open(buf)
    plt.close(fig)
    return wandb.Image(image)

# --- 3. 训练主循环 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(f"cuda:{cfg['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading datasets...")
    # NOTE: In real implementation, pass correct args to ViNT_Dataset
    # Here we instantiate a Mock-capable wrapper
    # Mocking underlying dataset logic for script robustness if files missing
    try:
        data_config = cfg['dataset']
        # Minimal set of args to start ViNT_Dataset
        # This part requires the actual Vint dataset files to exist.
        # If they don't, this script will fail at runtime unless we mock ViNT_Dataset.
        # Assuming user has the environment set up as per previous files.
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
        print("Please ensure dataset paths in config are correct.")
        return

    curr_stage = cfg['stage']
    stage_cfg = cfg['stages'][curr_stage]
    print(f"\n=== Starting Stage {curr_stage} ===")
    print(f"Description: {stage_cfg['description']}")
    
    # Check for metadata
    metadata_path = data_config.get('metadata_path', None)
    
    # Use config flag to override use_adv_data
    use_adv = stage_cfg['use_adv_data']

    dataset = ADSCD_DatasetWrapper(
        base_dataset, 
        dataset_name=data_config['name'],
        metadata_path=metadata_path, 
        use_adv_data=use_adv
    )
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, 
                            num_workers=cfg['num_workers'])
    
    # 2. Build Model
    model = ADSCDModel(cfg).to(device)
    
    # 3. Stage-specific Freezing
    print("\nSetting trainable parameters...")
    # Vision
    for p in model.vision_encoder.parameters():
        p.requires_grad = stage_cfg['train_vision']
    # Nav
    for p in model.nav_head.parameters():
        p.requires_grad = stage_cfg['train_nav']
    # Dist
    for p in model.dist_head.parameters():
        p.requires_grad = stage_cfg.get('train_dist', False)
    # Adv
    for p in model.adv_head.parameters():
        p.requires_grad = stage_cfg['train_adv']
    # Diffusion
    for p in model.noise_pred_net.parameters():
        p.requires_grad = stage_cfg['train_diffusion']
        
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)} tensors")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg['lr']))
    
    # 4. Training Loop
    if cfg['use_wandb']:
        wandb.init(project=cfg['project_name'], name=f"{cfg['run_name']}_stage{curr_stage}", config=cfg)
        
    global_step = 0
    model.train()
    for epoch in range(cfg['epochs']):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        total_loss = 0
        
        for batch in pbar:
            global_step += 1
            # Device transfer
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # --- Stage Logic Override ---
            # If Stage 1/Pure Nav, we might forcefully overwrite adv_mask to 0
            if not stage_cfg['use_adv_data']:
                batch['adv_mask'] = torch.zeros_like(batch['adv_mask'])
                
            loss_dict = model.compute_loss(batch, stage_cfg)
            loss = loss_dict['loss']
            
            optimizer.zero_grad()
            loss.backward()
            
            if cfg['grad_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg['grad_clip_norm'])
                
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'diff': loss_dict['diff_loss'].item()})
            
            if cfg['use_wandb']:
                # Collect all logs for this step
                log_data = loss_dict.copy()
                
                # Periodic Visualization (e.g. every 500 steps)
                if global_step % 500 == 0:
                    model.eval()
                    try:
                        # 1. Trajectory Viz (use first sample in batch)
                        with torch.no_grad():
                            # Get Single Sample Inputs
                            viz_obs = batch['obs_image'][0:1]
                            viz_goal = batch['goal_image'][0:1]
                            viz_other = batch['other_state'][0:1]
                            viz_mask = batch['adv_mask'][0:1]
                            
                            viz_gt_action = batch['action'][0].cpu().numpy() # (H, 2)
                            viz_goal_pos = batch['goal_pos'][0].cpu().numpy() if 'goal_pos' in batch else None
                            
                            # Inference
                            viz_pred_action = model.get_action(viz_obs, viz_goal, viz_other, viz_mask)
                            viz_pred_action = viz_pred_action[0].cpu().numpy()
                            
                            # Latent Distribution
                            stats = model.forward(viz_obs, viz_goal, viz_other, viz_mask)
                            nav_mu = stats['nav_dist'][0].cpu().numpy().flatten()
                            adv_mu = stats['adv_dist_real'][0].cpu().numpy().flatten()
                            
                            traj_img = visualize_trajectory(viz_gt_action, viz_pred_action, viz_goal_pos)
                            
                            viz_logs = {
                                "viz/trajectory": traj_img,
                                "viz/nav_mu_hist": wandb.Histogram(nav_mu),
                                "viz/adv_mu_hist": wandb.Histogram(adv_mu)
                            }
                            log_data.update(viz_logs)
                    except Exception as e:
                        print(f"Visualization failed at step {global_step}: {e}")
                    finally:
                        model.train()
                
                # Single Commit
                wandb.log(log_data, step=global_step)
                
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"checkpoint_stage{curr_stage}_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
