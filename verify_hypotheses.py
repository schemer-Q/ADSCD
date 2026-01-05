
import sys
import os
import pathlib
import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project roots to path
current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'diffusion_policy'))
sys.path.append(str(project_root / 'train'))

# Register OmegaConf resolver if not already registered
try:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception:
    pass

# Import SyntheticDataset from train_demo
from train_demo import SyntheticDataset

def load_model_hydra(checkpoint_path, device='cuda'):
    print(f"Loading model from {checkpoint_path} using Hydra...")
    
    # Load Configuration
    config_path = project_root / 'diffusion_policy' / 'diffusion_policy' / 'config'
    config_name = 'train_diffusion_hierarchical_workspace'
    
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = hydra.compose(config_name=config_name)
        
    # Instantiate Policy
    policy = hydra.utils.instantiate(cfg.policy)
    
    # Initialize normalizer
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['state_dict'])
    normalizer.load_state_dict(checkpoint['normalizer'])
    
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()
    return policy, cfg

def verify_hypotheses():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Model
    model_path = project_root / 'demo_model.ckpt'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run train_demo.py first.")
        return
        
    policy, cfg = load_model_hydra(model_path, device)
    
    # 2. Create Test Dataset
    print("\nGenerating Test Dataset...")
    # Use same params as training
    test_dataset = SyntheticDataset(
        size=50,
        horizon=cfg.horizon,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        other_state_dim=cfg.other_state_dim,
        context_size=cfg.vit_encoder.context_size
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    # --- Hypothesis 1: Gate Effectiveness (g=0 degradation) ---
    print("\n=== Verifying Hypothesis 1: Gate Effectiveness (g=0 degradation) ===")
    mse_g1_list_active = [] # For samples where GT g=1
    mse_g0_list_active = [] # For samples where GT g=1
    
    with torch.no_grad():
        for batch in test_loader:
            # Prepare inputs
            obs_img = batch['obs_img'].to(device)
            goal_img = batch['goal_img'].to(device)
            obs_lowdim = batch['obs'].to(device) # Use 'obs' which is lowdim in SyntheticDataset
            other_state = batch['other_state'].to(device)
            gt_action = batch['action'].to(device)
            gt_g = batch['adv_mask'].to(device) # [B, 1]
            
            # Run with g=1 (Default)
            obs_dict_g1 = {
                'obs': obs_lowdim, 
                'obs_img': obs_img,
                'goal_img': goal_img,
                'other_state': other_state,
                'adv_mask': torch.ones((obs_img.shape[0], 1), device=device)
            }
            result_g1 = policy.predict_action(obs_dict_g1)
            pred_action_g1 = result_g1['action']
            
            # Run with g=0 (Gate Closed)
            obs_dict_g0 = {
                'obs': obs_lowdim,
                'obs_img': obs_img,
                'goal_img': goal_img,
                'other_state': other_state,
                'adv_mask': torch.zeros((obs_img.shape[0], 1), device=device)
            }
            result_g0 = policy.predict_action(obs_dict_g0)
            pred_action_g0 = result_g0['action']
            
            # Debug shapes
            if len(mse_g1_list_active) == 0:
                print(f"Pred shape: {pred_action_g1.shape}")
                print(f"GT shape: {gt_action.shape}")

            # Align shapes if necessary
            T_pred = pred_action_g1.shape[1]
            T_gt = gt_action.shape[1]
            if T_pred != T_gt:
                min_T = min(T_pred, T_gt)
                pred_action_g1 = pred_action_g1[:, :min_T, :]
                pred_action_g0 = pred_action_g0[:, :min_T, :]
                gt_action = gt_action[:, :min_T, :]
            
            # Calculate MSE per sample
            mse_g1 = torch.mean((pred_action_g1 - gt_action)**2, dim=(1, 2)) # [B]
            mse_g0 = torch.mean((pred_action_g0 - gt_action)**2, dim=(1, 2)) # [B]
            
            # Filter for samples where GT g=1
            active_indices = (gt_g.squeeze() == 1).nonzero(as_tuple=True)[0]
            
            if len(active_indices) > 0:
                mse_g1_list_active.extend(mse_g1[active_indices].cpu().numpy())
                mse_g0_list_active.extend(mse_g0[active_indices].cpu().numpy())
            
    avg_mse_g1 = np.mean(mse_g1_list_active)
    avg_mse_g0 = np.mean(mse_g0_list_active)
    
    print(f"MSE on Adversarial Samples (GT g=1):")
    print(f"  With Gate Open (g=1):   {avg_mse_g1:.6f}")
    print(f"  With Gate Closed (g=0): {avg_mse_g0:.6f}")
    
    if avg_mse_g0 > avg_mse_g1:
        print("✅ Hypothesis 1 Verified: Performance degrades when adversarial gate is closed (g=0) for adversarial samples.")
    else:
        print("❌ Hypothesis 1 Failed: Performance did not degrade significantly with g=0.")
        
    # --- Hypothesis 2 & 3: Controllability & Stability ---
    print("\n=== Verifying Hypothesis 2 (Controllability) & 3 (Stability) ===")
    
    # Pick one sample from the dataset
    sample = test_dataset[0]
    # Add batch dim
    obs_img_sample = sample['obs_img'].unsqueeze(0).to(device)
    goal_img_sample = sample['goal_img'].unsqueeze(0).to(device)
    obs_lowdim_sample = sample['obs'].unsqueeze(0).to(device)
    
    # Create a batch with varying other_state
    batch_size = 10
    obs_img_batch = obs_img_sample.repeat(batch_size, 1, 1, 1)
    goal_img_batch = goal_img_sample.repeat(batch_size, 1, 1, 1)
    obs_lowdim_batch = obs_lowdim_sample.repeat(batch_size, 1, 1)
    
    # Vary other_state: linearly interpolate from -1 to 1
    other_state_batch = torch.zeros((batch_size, 2), device=device)
    other_state_batch[:, 0] = torch.linspace(-1, 1, batch_size) # Vary x
    other_state_batch[:, 1] = torch.linspace(-1, 1, batch_size) # Vary y
    
    # Run Encoder directly to check z_nav and z_adv
    with torch.no_grad():
        # Normalize obs_lowdim as done in policy
        nobs = policy.normalizer['obs'].normalize(obs_lowdim_batch)
        # Flatten: [B, T, D] -> [B, T*D] or just [B, D] if T=1. 
        # In policy.predict_action:
        # obs_lowdim = nobs[:,:self.n_obs_steps,:].reshape(nobs.shape[0], -1)
        obs_lowdim_flat = nobs[:, :policy.n_obs_steps, :].reshape(batch_size, -1)
        
        z_nav_mean, _, z_adv_mean, _ = policy.vit_encoder(
            obs_img_batch, 
            goal_img_batch, 
            adv_mask=torch.ones((batch_size, 1), device=device),
            obs_lowdim=obs_lowdim_flat,
            other_state=other_state_batch
        )
        
        # Run Policy to check action output
        obs_dict_batch = {
            'obs': obs_lowdim_batch,
            'obs_img': obs_img_batch,
            'goal_img': goal_img_batch,
            'other_state': other_state_batch,
            'adv_mask': torch.ones((batch_size, 1), device=device)
        }
        result_batch = policy.predict_action(obs_dict_batch)
        pred_actions = result_batch['action'] # [B, T, 2]
        
    # Analyze z_nav stability
    z_nav_std = torch.std(z_nav_mean, dim=0).mean().item()
    print(f"z_nav mean std across batch: {z_nav_std:.6f}")
    
    if z_nav_std < 1e-4:
        print("✅ Hypothesis 3 Verified: z_nav is stable (invariant to other_state).")
    else:
        print(f"❌ Hypothesis 3 Failed: z_nav varies with other_state (std={z_nav_std:.6f}).")
        
    # Analyze z_adv controllability
    z_adv_std = torch.std(z_adv_mean, dim=0).mean().item()
    print(f"z_adv mean std across batch: {z_adv_std:.6f}")
    
    if z_adv_std > 1e-4: # Lower threshold as latent space might be small
        print("✅ Hypothesis 2 (Part A) Verified: z_adv changes with other_state.")
    else:
        print(f"❌ Hypothesis 2 (Part A) Failed: z_adv is insensitive to other_state (std={z_adv_std:.6f}).")
        
    # Analyze Action controllability
    # Take the mean action across time steps for simplicity
    action_mean = pred_actions.mean(dim=1) # [B, 2]
    action_std = torch.std(action_mean, dim=0).mean().item()
    print(f"Action mean std across batch: {action_std:.6f}")
    
    if action_std > 1e-4:
        print("✅ Hypothesis 2 (Part B) Verified: Output action is controllable via other_state.")
    else:
        print(f"❌ Hypothesis 2 (Part B) Failed: Output action is insensitive to other_state (std={action_std:.6f}).")

if __name__ == "__main__":
    verify_hypotheses()
