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
def visualize_trajectory(gt_action, pred_action, title, filename):
    """Visualize and save trajectory comparison"""
    if torch.is_tensor(gt_action): gt_action = gt_action.cpu().numpy()
    if torch.is_tensor(pred_action): pred_action = pred_action.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    n_actions = gt_action.shape[-1]
    for i in range(n_actions):
        ax.plot(gt_action[:, i], label=f'GT Dim {i}', linestyle='-', alpha=0.5)
        ax.plot(pred_action[:, i], label=f'Pred Dim {i}', linestyle='--', linewidth=2)
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
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
    policy = hydra.utils.instantiate(cfg.policy)
    policy.to(device)
    policy.eval()
    
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
    
    # 3. Prepare Dummy Data
    batch_size = 1
    context_size = cfg.vit_encoder.context_size
    
    # Inputs for Encoders (E_nav, E_adv)
    obs_img = torch.randn(batch_size, 3*context_size, 128, 128, device=device) # Observation
    goal_img = torch.randn(batch_size, 3, 128, 128, device=device)             # Goal
    obs_lowdim = torch.randn(batch_size, cfg.horizon, cfg.obs_dim, device=device)
    
    # 4. Demo Scenarios
    
    # Scenario A: Non-Interactive / Safe (g=0)
    # README: "Force model to degrade to pure navigation when g=0"
    print("\n--- Scenario A: Pure Navigation (g=0) ---")
    g_safe = torch.zeros(batch_size, device=device) # g=0
    
    obs_dict_safe = {
        'obs': obs_lowdim,
        'obs_img': obs_img,
        'goal_img': goal_img,
        'adv_mask': g_safe
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
        'adv_mask': g_adv
    }
    
    with torch.no_grad():
        result_adv = policy.predict_action(obs_dict_adv)
        
    z_nav_adv = result_adv['z_nav_mean']
    z_adv_adv = result_adv['z_adv_mean']
    print(f"  z_nav norm: {torch.norm(z_nav_adv).item():.4f}")
    print(f"  z_adv norm: {torch.norm(z_adv_adv).item():.4f} (Active)")
    
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
    action_opt = result_opt['action'][0].cpu().numpy()
    
    mse_diff = np.mean((action_safe - action_adv)**2)
    print(f"  MSE (Safe vs Adv): {mse_diff:.6f}")
    
    output_dir = pathlib.Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    visualize_trajectory(action_safe, action_adv, "Safe (g=0) vs Adversarial (g=1)", output_dir / "comparison_g0_g1.png")
    visualize_trajectory(action_adv, action_opt, "Adversarial vs Optimized", output_dir / "comparison_adv_opt.png")
    
    print("\nDemo completed successfully.")
    print(f"Results saved to {output_dir.absolute()}")

if __name__ == "__main__":
    minimal_demo()
