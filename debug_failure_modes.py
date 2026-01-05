
import sys
import os
import pathlib
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

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
    
    if 'state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['state_dict'])
        if 'normalizer' in checkpoint:
            normalizer.load_state_dict(checkpoint['normalizer'])
        else:
            print("WARNING: 'normalizer' key not found in checkpoint. Using uninitialized normalizer.")
    else:
        print("WARNING: 'state_dict' key not found. Assuming checkpoint IS the state_dict.")
        policy.load_state_dict(checkpoint)
        print("WARNING: Cannot load normalizer from raw state dict.")

    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()
    return policy, cfg

def run_ideal_simulation(device, n_episodes=20, max_steps=30):
    """
    Hypothesis D: Test simulation with an ideal controller.
    Controller: action = (goal - curr) clipped to [-1, 1] (assuming normalized action space)
    """
    print("\n[Hypothesis D] Running Ideal Controller Simulation...")
    
    success_count = 0
    
    for i in range(n_episodes):
        curr_pos = np.random.randn(2)
        goal_pos = np.random.randn(2) * 2 + 2
        
        traj = [curr_pos.copy()]
        
        for step in range(max_steps):
            # Ideal Action: Move towards goal
            diff = goal_pos - curr_pos
            # Normalize direction
            dist = np.linalg.norm(diff)
            if dist > 1e-6:
                direction = diff / dist
            else:
                direction = np.zeros(2)
            
            # Assume max speed is 1.0 (in action space)
            action = direction * 1.0
            
            # Simulation Step (same as comprehensive_evaluation.py)
            # curr_pos = curr_pos + action * 0.5
            curr_pos = curr_pos + action * 0.5
            traj.append(curr_pos.copy())
            
            if np.linalg.norm(curr_pos - goal_pos) < 0.5:
                success_count += 1
                break
                
    success_rate = success_count / n_episodes
    print(f"  Ideal Controller Success Rate: {success_rate*100:.1f}%")
    if success_rate > 0.9:
        print("  -> Simulation Environment is likely OK.")
    else:
        print("  -> Simulation Environment might be BROKEN (or max_steps too low).")
    return success_rate

def analyze_action_stats(policy, device, n_samples=100):
    """
    Hypothesis A & C: Analyze action magnitude and normalization.
    """
    print("\n[Hypothesis A & C] Analyzing Action Statistics...")
    
    # Create dummy inputs
    obs_img = torch.randn(n_samples, 12, 384, 384).to(device)
    goal_img = torch.randn(n_samples, 3, 384, 384).to(device)
    other_state = torch.randn(n_samples, 2).to(device)
    
    # Dummy lowdim obs (padded)
    obs_padded = torch.zeros(n_samples, 4, 20).to(device)
    
    # 1. Check Normalizer
    print("  Normalizer Stats:")
    normalizer = policy.normalizer
    if 'action' in normalizer.params_dict:
        stats = normalizer.params_dict['action']
        print(f"    Scale: {stats['scale'].mean().item():.4f} (Mean across dims)")
        print(f"    Offset: {stats['offset'].mean().item():.4f} (Mean across dims)")
        print(f"    Min Scale: {stats['scale'].min().item():.4f}")
        print(f"    Max Scale: {stats['scale'].max().item():.4f}")
    else:
        print("    No 'action' key in normalizer params.")

    # 2. Check Output Magnitude for g=0 vs g=1
    modes = [0.0, 1.0]
    for g_val in modes:
        mode_name = "Nav (g=0)" if g_val == 0.0 else "Adv (g=1)"
        adv_mask = torch.full((n_samples, 1), g_val).to(device)
        
        obs_dict = {
            'obs': obs_padded,
            'obs_img': obs_img,
            'goal_img': goal_img,
            'other_state': other_state,
            'adv_mask': adv_mask
        }
        
        with torch.no_grad():
            result = policy.predict_action(obs_dict)
            # result['action'] is [B, T, D]
            actions = result['action'][:, 0, :].cpu().numpy() # First step
            
        # Stats
        mag = np.linalg.norm(actions, axis=1)
        print(f"\n  {mode_name} Action Stats:")
        print(f"    Mean Magnitude: {np.mean(mag):.4f}")
        print(f"    Max Magnitude:  {np.max(mag):.4f}")
        print(f"    Min Magnitude:  {np.min(mag):.4f}")
        print(f"    Mean Action Values: {np.mean(actions, axis=0)}")
        
        if np.mean(mag) < 0.1:
            print(f"    -> WARNING: Actions are very small! Hypothesis A might be true.")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    checkpoint_path = project_root / 'demo_model.ckpt'
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    try:
        policy, cfg = load_model_hydra(str(checkpoint_path), device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Run Tests
    run_ideal_simulation(device)
    analyze_action_stats(policy, device)

if __name__ == "__main__":
    main()
