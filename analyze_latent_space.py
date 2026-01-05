
import sys
import os
import pathlib
import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

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

def analyze_latent_space():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Model
    model_path = project_root / 'demo_model.ckpt'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run train_demo.py first.")
        return
        
    policy, cfg = load_model_hydra(model_path, device)
    
    # 2. Generate Analysis Dataset (Larger than verification)
    print("\nGenerating Analysis Dataset (N=200)...")
    dataset = SyntheticDataset(
        size=200,
        horizon=cfg.horizon,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        other_state_dim=cfg.other_state_dim,
        context_size=cfg.vit_encoder.context_size
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)
    
    # Collect Latents and Metadata
    z_nav_list = []
    z_adv_list = []
    g_list = []
    goal_dist_list = []
    other_dist_list = []
    
    print("Extracting Latents...")
    with torch.no_grad():
        for batch in loader:
            obs_img = batch['obs_img'].to(device)
            goal_img = batch['goal_img'].to(device)
            obs_lowdim = batch['obs'].to(device)
            other_state = batch['other_state'].to(device)
            g = batch['adv_mask'].to(device)
            
            # Normalize obs
            nobs = policy.normalizer['obs'].normalize(obs_lowdim)
            obs_lowdim_flat = nobs[:, :policy.n_obs_steps, :].reshape(obs_lowdim.shape[0], -1)
            
            # Encode
            z_nav_mean, _, z_adv_mean, _ = policy.vit_encoder(
                obs_img, 
                goal_img, 
                adv_mask=g, # Mask doesn't affect mean output in forward, only sampling
                obs_lowdim=obs_lowdim_flat,
                other_state=other_state
            )
            
            z_nav_list.append(z_nav_mean.cpu().numpy())
            z_adv_list.append(z_adv_mean.cpu().numpy())
            g_list.append(g.cpu().numpy())
            
            # Calculate distances (Metadata)
            # obs_lowdim is [B, T, 2]. Current pos is at T=0 (or T=obs_horizon-1 depending on convention)
            # SyntheticDataset: obs[:, :2] = current_pos. 
            # Let's assume obs[:, 0, :2] is current pos.
            curr_pos = obs_lowdim[:, 0, :2].cpu().numpy()
            
            # Goal pos is not directly in batch, but we can infer it or just use the fact that 
            # in SyntheticDataset, action ~ (goal - curr). 
            # But wait, SyntheticDataset doesn't return goal_pos, only goal_img.
            # However, we can cheat and look at the 'action' to guess goal direction?
            # Or better, let's modify SyntheticDataset to return metadata? 
            # For now, let's assume we can't modify dataset easily without breaking things.
            # Wait, I can modify SyntheticDataset in train_demo.py? No, I imported it.
            # I'll just use the 'action' (GT) as a proxy for goal direction for nav.
            # GT Action ~ (Goal - Curr) + g*(Curr - Other).
            # If g=0, Action ~ Goal - Curr.
            # So for g=0 samples, action magnitude is proportional to goal distance.
            
            # Actually, let's just use the latent analysis on the variables we have.
            # We have 'other_state' which is 'other_pos' in SyntheticDataset.
            # We don't have 'goal_pos'.
            # But we can check correlation with 'other_state' magnitude.
            
            # Let's try to reconstruct goal_dist from action for g=0 samples.
            gt_action = batch['action'][:, 0, :].cpu().numpy() # [B, 2]
            # For g=0, action is roughly goal_vector.
            # For g=1, it's mixed.
            
            # Let's use 'other_state' magnitude as 'other_dist' (assuming other_state is relative pos? 
            # In SyntheticDataset: other_state = other_pos. obs = current_pos.
            # So dist = norm(current_pos - other_pos).
            other_pos = other_state.cpu().numpy()
            dist_to_other = np.linalg.norm(curr_pos - other_pos, axis=1)
            other_dist_list.append(dist_to_other)
            
            # For goal distance, we only have it reliably if we knew goal_pos.
            # Let's skip exact goal distance correlation if we can't get it, 
            # OR we can instantiate a dataset that returns it.
            # I will define a local dataset class that returns metadata.
            
    # Concatenate
    z_nav = np.concatenate(z_nav_list, axis=0)
    z_adv = np.concatenate(z_adv_list, axis=0)
    g = np.concatenate(g_list, axis=0).squeeze()
    other_dist = np.concatenate(other_dist_list, axis=0)
    
    # --- Analysis 1: z_adv Distribution (g=0 vs g=1) ---
    print("\n--- Analysis 1: z_adv Distribution ---")
    # Since g is random, we expect distributions to be similar if encoder is unbiased.
    z_adv_g0 = z_adv[g == 0]
    z_adv_g1 = z_adv[g == 1]
    
    print(f"z_adv (g=0) mean norm: {np.linalg.norm(z_adv_g0, axis=1).mean():.4f}")
    print(f"z_adv (g=1) mean norm: {np.linalg.norm(z_adv_g1, axis=1).mean():.4f}")
    
    # --- Analysis 2: z_nav vs Goal (using Sensitivity Analysis instead of Correlation) ---
    # Since we don't have explicit goal pos in the loaded dataset, we'll do a sensitivity test.
    print("\n--- Analysis 2 & 3: Disentanglement Quantification (Sensitivity Analysis) ---")
    
    # We will generate a synthetic batch where we control variables.
    # Base case
    B = 50
    obs_img = torch.randn(B, 12, 384, 384).to(device) # Dummy images
    goal_img = torch.randn(B, 3, 384, 384).to(device)
    
    # Variable 1: Goal (Simulated by changing goal_img)
    # Variable 2: Other State (Simulated by changing other_state)
    
    # We need to see if z_nav changes when goal_img changes (Should be High)
    # We need to see if z_nav changes when other_state changes (Should be Low)
    
    # We need to see if z_adv changes when goal_img changes (Should be Low)
    # We need to see if z_adv changes when other_state changes (Should be High)
    
    # 1. Vary Goal
    goal_img_var = torch.randn(B, 3, 384, 384).to(device)
    other_state_fixed = torch.zeros(B, 2).to(device)
    # obs_lowdim_fixed must match the dimension expected by the encoder
    # In config: obs_lowdim_dim: ${eval:'${n_obs_steps}*${obs_dim}'}
    # n_obs_steps=2, obs_dim=20 (from SyntheticDataset default) -> 40?
    # Wait, SyntheticDataset default obs_dim=20.
    # But in train_demo.py, we might have used different values?
    # Let's check cfg.obs_dim
    obs_lowdim_dim = cfg.n_obs_steps * cfg.obs_dim
    obs_lowdim_fixed = torch.zeros(B, obs_lowdim_dim).to(device)
    
    with torch.no_grad():
        z_nav_goal_var, _, z_adv_goal_var, _ = policy.vit_encoder(
            obs_img, goal_img_var, adv_mask=None, obs_lowdim=obs_lowdim_fixed, other_state=other_state_fixed
        )
    
    # 2. Vary Other State
    goal_img_fixed = goal_img[0:1].repeat(B, 1, 1, 1)
    other_state_var = torch.randn(B, 2).to(device)
    
    with torch.no_grad():
        z_nav_other_var, _, z_adv_other_var, _ = policy.vit_encoder(
            obs_img, goal_img_fixed, adv_mask=None, obs_lowdim=obs_lowdim_fixed, other_state=other_state_var
        )
        
    # Calculate Variances
    var_z_nav_wrt_goal = torch.var(z_nav_goal_var, dim=0).mean().item()
    var_z_nav_wrt_other = torch.var(z_nav_other_var, dim=0).mean().item()
    
    var_z_adv_wrt_goal = torch.var(z_adv_goal_var, dim=0).mean().item()
    var_z_adv_wrt_other = torch.var(z_adv_other_var, dim=0).mean().item()
    
    print(f"Var(z_nav | Delta Goal):  {var_z_nav_wrt_goal:.6f} (Expected: High)")
    print(f"Var(z_nav | Delta Other): {var_z_nav_wrt_other:.6f} (Expected: Low/Zero)")
    
    print(f"Var(z_adv | Delta Goal):  {var_z_adv_wrt_goal:.6f} (Expected: Low/Zero)")
    print(f"Var(z_adv | Delta Other): {var_z_adv_wrt_other:.6f} (Expected: High)")
    
    # Metrics
    # Avoid division by zero
    eps = 1e-8
    nav_disentanglement = 1.0 - (var_z_nav_wrt_other / (var_z_nav_wrt_goal + eps))
    adv_disentanglement = 1.0 - (var_z_adv_wrt_goal / (var_z_adv_wrt_other + eps))
    
    print(f"\nQuantified Disentanglement Scores (1.0 is perfect):")
    print(f"Nav Intent Disentanglement: {nav_disentanglement:.4f}")
    print(f"Adv Intent Disentanglement: {adv_disentanglement:.4f}")
    
    # --- Visualization ---
    print("\nGenerating Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: z_adv Distribution
    axes[0].hist(np.linalg.norm(z_adv_g0, axis=1), bins=20, alpha=0.5, label='g=0', color='red', density=True)
    axes[0].hist(np.linalg.norm(z_adv_g1, axis=1), bins=20, alpha=0.5, label='g=1', color='green', density=True)
    axes[0].set_title('Analysis 1: z_adv Norm Distribution')
    axes[0].set_xlabel('|z_adv|')
    axes[0].legend()
    
    # Plot 2: PCA of z_adv colored by Other Distance
    pca = PCA(n_components=2)
    z_adv_pca = pca.fit_transform(z_adv)
    sc = axes[1].scatter(z_adv_pca[:, 0], z_adv_pca[:, 1], c=other_dist, cmap='viridis')
    plt.colorbar(sc, ax=axes[1], label='Distance to Other')
    axes[1].set_title('Analysis 2: z_adv PCA (Semantics)')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    
    # Plot 3: Disentanglement Bar Chart
    metrics = ['Nav Disentanglement', 'Adv Disentanglement']
    scores = [nav_disentanglement, adv_disentanglement]
    axes[2].bar(metrics, scores, color=['blue', 'orange'])
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title('Analysis 3: Disentanglement Scores')
    for i, v in enumerate(scores):
        axes[2].text(i, v + 0.02, f"{v:.4f}", ha='center')
        
    plt.tight_layout()
    plt.savefig(project_root / 'latent_space_analysis.png')
    print(f"Analysis plot saved to {project_root / 'latent_space_analysis.png'}")

if __name__ == "__main__":
    analyze_latent_space()
