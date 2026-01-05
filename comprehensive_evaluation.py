
import sys
import os
import pathlib
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project roots to path
current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'diffusion_policy'))
sys.path.append(str(project_root / 'train'))

# Register OmegaConf resolver
try:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception:
    pass

def load_model_hydra(checkpoint_path, device='cuda'):
    print(f"Loading model from {checkpoint_path}...")
    config_path = project_root / 'diffusion_policy' / 'diffusion_policy' / 'config'
    config_name = 'train_diffusion_hierarchical_workspace'
    
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = hydra.compose(config_name=config_name)
        
    policy = hydra.utils.instantiate(cfg.policy)
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['state_dict'])
    normalizer.load_state_dict(checkpoint['normalizer'])
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()
    return policy, cfg

def compute_robust_disentanglement(policy, device, n_samples=100):
    print("\n=== 1. Robust Disentanglement Analysis ===")
    
    # Base inputs
    obs_img = torch.randn(n_samples, 12, 384, 384).to(device)
    
    # 1. Analyze z_nav sensitivity
    # We want z_nav to be sensitive to Goal, insensitive to Other
    print("Analyzing z_nav sensitivity...")
    
    # Case A: Vary Goal, Fix Other
    goal_A1 = torch.randn(n_samples, 3, 384, 384).to(device)
    goal_A2 = torch.randn(n_samples, 3, 384, 384).to(device)
    other_fixed = torch.randn(n_samples, 2).to(device)
    obs_lowdim = torch.zeros(n_samples, 40).to(device) # Dummy lowdim
    
    with torch.no_grad():
        z_nav_A1, _, _, _ = policy.vit_encoder(obs_img, goal_A1, obs_lowdim=obs_lowdim, other_state=other_fixed)
        z_nav_A2, _, _, _ = policy.vit_encoder(obs_img, goal_A2, obs_lowdim=obs_lowdim, other_state=other_fixed)
    
    diff_goal = torch.norm(z_nav_A1 - z_nav_A2, dim=1).cpu().numpy()
    median_diff_goal = np.median(diff_goal)
    
    # Case B: Fix Goal, Vary Other
    goal_fixed = goal_A1
    other_B1 = torch.randn(n_samples, 2).to(device)
    other_B2 = torch.randn(n_samples, 2).to(device)
    
    with torch.no_grad():
        z_nav_B1, _, _, _ = policy.vit_encoder(obs_img, goal_fixed, obs_lowdim=obs_lowdim, other_state=other_B1)
        z_nav_B2, _, _, _ = policy.vit_encoder(obs_img, goal_fixed, obs_lowdim=obs_lowdim, other_state=other_B2)
        
    diff_other = torch.norm(z_nav_B1 - z_nav_B2, dim=1).cpu().numpy()
    median_diff_other = np.median(diff_other)
    
    # Metric
    eps = 1e-6
    nav_score = median_diff_goal / (median_diff_goal + median_diff_other + eps)
    print(f"z_nav Sensitivity to Goal (Median Diff): {median_diff_goal:.6f}")
    print(f"z_nav Sensitivity to Other (Median Diff): {median_diff_other:.6f}")
    print(f"-> Nav Disentanglement Score: {nav_score:.4f} (Target: 1.0)")
    
    # 2. Analyze z_adv sensitivity
    # We want z_adv to be sensitive to Other, insensitive to Goal
    print("\nAnalyzing z_adv sensitivity...")
    
    with torch.no_grad():
        # Vary Goal (Reuse A1, A2)
        _, _, z_adv_A1, _ = policy.vit_encoder(obs_img, goal_A1, obs_lowdim=obs_lowdim, other_state=other_fixed)
        _, _, z_adv_A2, _ = policy.vit_encoder(obs_img, goal_A2, obs_lowdim=obs_lowdim, other_state=other_fixed)
        
        # Vary Other (Reuse B1, B2)
        _, _, z_adv_B1, _ = policy.vit_encoder(obs_img, goal_fixed, obs_lowdim=obs_lowdim, other_state=other_B1)
        _, _, z_adv_B2, _ = policy.vit_encoder(obs_img, goal_fixed, obs_lowdim=obs_lowdim, other_state=other_B2)
        
    diff_goal_adv = torch.norm(z_adv_A1 - z_adv_A2, dim=1).cpu().numpy()
    median_diff_goal_adv = np.median(diff_goal_adv)
    
    diff_other_adv = torch.norm(z_adv_B1 - z_adv_B2, dim=1).cpu().numpy()
    median_diff_other_adv = np.median(diff_other_adv)
    
    adv_score = median_diff_other_adv / (median_diff_goal_adv + median_diff_other_adv + eps)
    print(f"z_adv Sensitivity to Goal (Median Diff): {median_diff_goal_adv:.6f}")
    print(f"z_adv Sensitivity to Other (Median Diff): {median_diff_other_adv:.6f}")
    print(f"-> Adv Disentanglement Score: {adv_score:.4f} (Target: 1.0)")
    
    return nav_score, adv_score

def run_simulation(policy, device, mode='nav', n_episodes=20, max_steps=30):
    """
    Simple kinematic simulation.
    mode: 'nav' (g=0) or 'adv' (g=1)
    """
    print(f"\nRunning Simulation (Mode: {mode})...")
    
    success_count = 0
    min_dists_to_other = []
    path_lengths = []
    
    all_trajectories = []
    all_goals = []
    all_others = []
    
    for i in range(n_episodes):
        # Init Episode
        curr_pos = np.random.randn(2) # Start around 0
        goal_pos = np.random.randn(2) * 2 + 2 # Goal further away
        other_pos = (curr_pos + goal_pos) / 2.0 + np.random.randn(2) * 0.2 # Other in between
        
        all_goals.append(goal_pos)
        all_others.append(other_pos)
        
        # Convert to tensors
        goal_img = torch.randn(1, 3, 384, 384).to(device) # Dummy goal img
        obs_img = torch.randn(1, 12, 384, 384).to(device) # Dummy obs img
        
        traj = [curr_pos.copy()]
        
        for step in range(max_steps):
            # Prepare Obs
            # obs_lowdim needs to match training format. 
            # In SyntheticDataset, obs[:, :2] is current pos.
            # We need to construct a history of obs. For simplicity, repeat current.
            obs_hist = np.tile(curr_pos, (4, 1)) # [4, 2]
            # But wait, config expects flattened obs of size 40 (20*2)?
            # SyntheticDataset obs_dim=20.
            # We need to pad.
            obs_padded = np.zeros((4, 20))
            obs_padded[:, :2] = obs_hist
            
            obs_tensor = torch.from_numpy(obs_padded).float().unsqueeze(0).to(device) # [1, 4, 20]
            other_tensor = torch.from_numpy(other_pos).float().unsqueeze(0).to(device) # [1, 2]
            
            # Gate
            g_val = 1.0 if mode == 'adv' else 0.0
            adv_mask = torch.tensor([[g_val]]).to(device)
            
            # Predict
            obs_dict = {
                'obs': obs_tensor,
                'obs_img': obs_img,
                'goal_img': goal_img,
                'other_state': other_tensor,
                'adv_mask': adv_mask
            }
            
            with torch.no_grad():
                result = policy.predict_action(obs_dict)
                action = result['action'][0, 0, :].cpu().numpy() # Take first step action [2]
            
            # Step
            # Action in SyntheticDataset was (Goal - Curr) + ...
            # So it's a velocity/displacement vector.
            curr_pos = curr_pos + action * 0.5 # Scale down for stability
            traj.append(curr_pos.copy())
            
            # Check Goal
            dist_to_goal = np.linalg.norm(curr_pos - goal_pos)
            if dist_to_goal < 0.5:
                success_count += 1
                break
        
        traj = np.array(traj)
        all_trajectories.append(traj)
        
        # Metrics
        dists_to_other = np.linalg.norm(traj - other_pos, axis=1)
        min_dists_to_other.append(np.min(dists_to_other))
        
        path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        path_lengths.append(path_len)
        
    success_rate = success_count / n_episodes
    avg_min_dist = np.mean(min_dists_to_other)
    avg_path_len = np.mean(path_lengths)
    
    print(f"  Success Rate: {success_rate*100:.1f}%")
    print(f"  Avg Min Dist to Other: {avg_min_dist:.4f}")
    print(f"  Avg Path Length: {avg_path_len:.4f}")
    
    return success_rate, avg_min_dist, avg_path_len, all_trajectories, all_goals, all_others

def comprehensive_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model_path = project_root / 'demo_model.ckpt'
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    policy, cfg = load_model_hydra(model_path, device)
    
    # 1. Disentanglement
    nav_score, adv_score = compute_robust_disentanglement(policy, device)
    
    # 2. Performance Benchmark
    print("\n=== 2. Performance Benchmark ===")
    # Run Nav Mode (g=0)
    sr_nav, dist_nav, len_nav, traj_nav, goals_nav, others_nav = run_simulation(policy, device, mode='nav')
    
    # Run Adv Mode (g=1)
    sr_adv, dist_adv, len_adv, traj_adv, goals_adv, others_adv = run_simulation(policy, device, mode='adv')
    
    print("\n=== Benchmark Summary ===")
    print(f"Nav Mode (g=0): Success={sr_nav*100:.1f}%, MinDistToOther={dist_nav:.4f}")
    print(f"Adv Mode (g=1): Success={sr_adv*100:.1f}%, MinDistToOther={dist_adv:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    print("Note: Adversarial Mode (g=1) is defined as generating interference/risky trajectories (Aggressive Interaction).")
    
    if dist_adv < dist_nav:
        print("✅ Adversarial mode (g=1) results in closer proximity (potential interference) compared to Nav mode.")
        print(f"   -> Distance reduced by {dist_nav - dist_adv:.4f}")
    else:
        print("❓ Adversarial mode did not decrease distance to opponent.")
        print("   -> It might be behaving passively or the random inputs are dominating.")
        
    if abs(sr_nav - sr_adv) < 0.2:
        print("✅ Goal reaching capability is maintained in Adversarial mode.")
    else:
        print("⚠️ Goal reaching capability degraded in Adversarial mode.")

    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Disentanglement
    ax[0].bar(['Nav Intent', 'Adv Intent'], [nav_score, adv_score], color=['blue', 'orange'])
    ax[0].set_ylim(0, 1.1)
    ax[0].set_title('Robust Disentanglement Scores')
    ax[0].set_ylabel('Relative Sensitivity Score')
    
    # Plot 2: Performance
    x = np.arange(2)
    width = 0.35
    ax[1].bar(x - width/2, [dist_nav, dist_adv], width, label='Min Dist to Other', color='purple')
    ax[1].bar(x + width/2, [sr_nav, sr_adv], width, label='Success Rate', color='green')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(['Nav Mode (g=0)', 'Adv Mode (g=1)'])
    ax[1].set_title('Performance Comparison')
    ax[1].legend()

    # Plot 3: Trajectories
    ax[2].set_title('Trajectories (Nav=Blue, Adv=Orange)')
    # Plot Nav
    for i, traj in enumerate(traj_nav):
        ax[2].plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.3, label='Nav' if i==0 else "")
        ax[2].scatter(traj[0, 0], traj[0, 1], color='green', marker='o', s=20, label='Start' if i==0 else "")
        ax[2].scatter(goals_nav[i][0], goals_nav[i][1], color='red', marker='*', s=50, label='Goal' if i==0 else "")
        ax[2].scatter(others_nav[i][0], others_nav[i][1], color='black', marker='x', s=30, label='Other' if i==0 else "")
    
    # Plot Adv
    for i, traj in enumerate(traj_adv):
        ax[2].plot(traj[:, 0], traj[:, 1], color='orange', alpha=0.3, label='Adv' if i==0 else "")
    
    ax[2].legend()
    ax[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(project_root / 'comprehensive_benchmark.png')
    print(f"Benchmark plot saved to {project_root / 'comprehensive_benchmark.png'}")
    
    plt.tight_layout()
    plt.savefig(project_root / 'comprehensive_benchmark.png')
    print(f"Benchmark plot saved to {project_root / 'comprehensive_benchmark.png'}")

if __name__ == "__main__":
    comprehensive_benchmark()
