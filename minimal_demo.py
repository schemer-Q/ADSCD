import sys
import os
import pathlib
import torch
import hydra
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

# --- Environment Setup ---
# Fix for torch.xpu and torch.mps on some systems
for backend in ['xpu', 'mps', 'vulkan', 'rocm']:
    if not hasattr(torch, backend):
        class DummyBackend:
            @staticmethod
            def empty_cache(): pass
            @staticmethod
            def current_stream(): return None
            @staticmethod
            def synchronize(): pass
        setattr(torch, backend, DummyBackend)
        setattr(getattr(torch, backend), 'empty_cache', lambda: None)

# Add project roots to path
current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'diffusion_policy'))
sys.path.append(str(project_root / 'train'))

# Register OmegaConf resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# --- Helper Functions ---
def visualize_trajectory(traj1, traj2, label1, label2, title, filename):
    """Visualize and save trajectory comparison"""
    if torch.is_tensor(traj1): traj1 = traj1.cpu().numpy()
    if torch.is_tensor(traj2): traj2 = traj2.cpu().numpy()
    
    n_actions = traj1.shape[-1]
    # Create subplots for each dimension to make it clearer
    fig, axes = plt.subplots(n_actions, 1, figsize=(10, 3 * n_actions), sharex=True)
    if n_actions == 1:
        axes = [axes]
    
    for i in range(n_actions):
        ax = axes[i]
        ax.plot(traj1[:, i], label=f'{label1}', linestyle='-', linewidth=2, alpha=0.7)
        ax.plot(traj2[:, i], label=f'{label2}', linestyle='--', linewidth=2, alpha=0.9)
        ax.set_ylabel(f'Action Dim {i}')
        ax.legend()
        ax.grid(True)
    
    axes[-1].set_xlabel('Time Step')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization to {filename}")

def minimal_demo():
    print("=== Adversarial Intent Factorization: Minimal Demo ===")
    
    # 1. Load Configuration
    # We use the existing config for hierarchical diffusion
    config_path = project_root / 'diffusion_policy' / 'diffusion_policy' / 'config'
    config_name = 'train_diffusion_hierarchical_workspace'
    
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = hydra.compose(config_name=config_name)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Initialize Model Components
    # This corresponds to the architecture in README:
    # - E_nav & E_adv: Implemented in ViTHierarchicalEncoder
    # - D_diff: Implemented in DiffusionHierarchicalPolicy (ConditionalUnet1D)
    print("\nInitializing Model Framework...")
    # Speed up: Reduce inference steps for demo
    if 'policy' in cfg and 'num_inference_steps' in cfg.policy:
        cfg.policy.num_inference_steps = 20
        print(f"  [SpeedUp] Reduced inference steps to {cfg.policy.num_inference_steps}")
    
    policy = hydra.utils.instantiate(cfg.policy)
    policy.to(device)
    policy.eval()
    
    # Ensure policy uses the reduced steps if not set by config
    if hasattr(policy, 'num_inference_steps'):
        policy.num_inference_steps = 20
    
    # Initialize normalizer (required for policy operation)
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    # Fit with dummy data to set statistics
    dummy_batch = {
        'obs': torch.randn(2, cfg.horizon, cfg.obs_dim, device=device),
        'action': torch.randn(2, cfg.horizon, cfg.action_dim, device=device)
    }
    normalizer.fit(dummy_batch)
    policy.set_normalizer(normalizer)
    
    print("Model initialized.")
    print(f"  Latent Nav Dim (z_nav): {cfg.latent_dim_nav}")
    print(f"  Latent Adv Dim (z_adv): {cfg.latent_dim_adv}")
    print(f"  Other State Dim: {cfg.other_state_dim}")
    
    # 3. Prepare Dummy Data
    batch_size = 1
    context_size = cfg.vit_encoder.context_size
    
    # Inputs for Encoders (E_nav, E_adv)
    obs_img = torch.randn(batch_size, 3*context_size, 128, 128, device=device) # Observation
    goal_img = torch.randn(batch_size, 3, 128, 128, device=device)             # Goal
    obs_lowdim = torch.randn(batch_size, cfg.horizon, cfg.obs_dim, device=device)
    other_state = torch.randn(batch_size, cfg.other_state_dim, device=device)  # Other agent state
    
    # 4. Demo Scenarios
    
    # Scenario A: Non-Interactive / Safe (g=0)
    # README: "Force model to degrade to pure navigation when g=0"
    print("\n--- Scenario A: Pure Navigation (g=0) ---")
    g_safe = torch.zeros(batch_size, device=device) # g=0
    
    obs_dict_safe = {
        'obs': obs_lowdim,
        'obs_img': obs_img,
        'goal_img': goal_img,
        'adv_mask': g_safe,
        'other_state': other_state
    }
    
    with torch.no_grad():
        result_safe = policy.predict_action(obs_dict_safe)
    
    z_nav_safe = result_safe['z_nav_mean']
    z_adv_safe = result_safe['z_adv_mean']
    print(f"  z_nav norm: {torch.norm(z_nav_safe).item():.4f}")
    print(f"  z_adv norm: {torch.norm(z_adv_safe).item():.4f} (Should be close to 0 or irrelevant)")
    
    # Scenario B: Interactive / Adversarial (g=1)
    # README: "z_adv only activated in g=1 samples"
    print("\n--- Scenario B: Adversarial Interaction (g=1) ---")
    g_adv = torch.ones(batch_size, device=device) # g=1
    
    obs_dict_adv = {
        'obs': obs_lowdim,
        'obs_img': obs_img,
        'goal_img': goal_img,
        'adv_mask': g_adv,
        'other_state': other_state
    }
    
    with torch.no_grad():
        result_adv = policy.predict_action(obs_dict_adv)
        
    z_nav_adv = result_adv['z_nav_mean']
    z_adv_adv = result_adv['z_adv_mean']
    print(f"  z_nav norm: {torch.norm(z_nav_adv).item():.4f}")
    print(f"  z_adv norm: {torch.norm(z_adv_adv).item():.4f} (Active)")

    # Scenario B2: Verify other_state influence
    print("\n--- Scenario B2: Verify other_state influence ---")
    # Change other_state
    other_state_new = torch.randn(batch_size, cfg.other_state_dim, device=device)
    obs_dict_adv_new = {
        'obs': obs_lowdim,
        'obs_img': obs_img,
        'goal_img': goal_img,
        'adv_mask': g_adv,
        'other_state': other_state_new
    }
    with torch.no_grad():
        result_adv_new = policy.predict_action(obs_dict_adv_new)
    
    z_nav_new = result_adv_new['z_nav_mean']
    z_adv_new = result_adv_new['z_adv_mean']
    
    # Check if z_nav changed (should NOT change)
    nav_diff = torch.norm(z_nav_adv - z_nav_new).item()
    print(f"  z_nav difference (should be 0): {nav_diff:.6f}")
    
    # Check if z_adv changed (SHOULD change)
    adv_diff = torch.norm(z_adv_adv - z_adv_new).item()
    print(f"  z_adv difference (should be > 0): {adv_diff:.6f}")
    
    # Scenario C: Adversarial Optimization
    # README: "Fix z_nav, optimize z_adv"
    print("\n--- Scenario C: Adversarial Optimization ---")
    # We want to find a z_adv that maximizes some objective (here simulated by the optimize method)
    # Note: The policy class needs to implement optimize_adversarial_intent
    if hasattr(policy, 'optimize_adversarial_intent'):
        with torch.no_grad(): # The method handles gradients internally if needed
             result_opt = policy.optimize_adversarial_intent(obs_dict_adv, num_steps=10, lr=0.1)
        print("  Optimization complete.")
        print(f"  Original z_adv (from Scenario B): {z_adv_adv[0, :5].cpu().numpy()}...")
        print(f"  Optimized z_adv: {result_opt['z_adv'][0, :5].cpu().numpy()}...")
    else:
        print("  optimize_adversarial_intent method not found in policy.")
        result_opt = result_adv # Fallback

    # 5. Visualization & Comparison
    print("\n--- Results Comparison ---")
    action_safe = result_safe['action'][0].cpu().numpy()
    action_adv = result_adv['action'][0].cpu().numpy()
    action_adv_new = result_adv_new['action'][0].cpu().numpy()
    action_opt = result_opt['action'][0].cpu().numpy()
    
    mse_diff = np.mean((action_safe - action_adv)**2)
    print(f"  MSE (Safe vs Adv): {mse_diff:.6f}")
    
    output_dir = pathlib.Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    visualize_trajectory(action_safe, action_adv, "Safe (g=0)", "Adversarial (g=1)", "Safe vs Adversarial Behavior", output_dir / "comparison_g0_g1.png")
    visualize_trajectory(action_adv, action_adv_new, "Adversarial (State A)", "Adversarial (State B)", "Impact of Changing Other State", output_dir / "comparison_other_state.png")
    visualize_trajectory(action_adv, action_opt, "Adversarial (Initial)", "Adversarial (Optimized)", "Adversarial Optimization", output_dir / "comparison_adv_opt.png")
    
    print("\nDemo completed successfully.")
    print(f"Results saved to {output_dir.absolute()}")

if __name__ == "__main__":
    minimal_demo()
