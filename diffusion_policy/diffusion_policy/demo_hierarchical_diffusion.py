# 修复torch.xpu问题
import torch
if not hasattr(torch, 'xpu'):
    class DummyXPU:
        @staticmethod
        def empty_cache():
            pass
    
    torch.xpu = DummyXPU
    torch.xpu.empty_cache = lambda: None
import sys
import os
# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import hydra
from omegaconf import OmegaConf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from diffusion_policy.policy.diffusion_hierarchical_policy import DiffusionHierarchicalPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)

def visualize_trajectory(gt_action, pred_action, title):
    """
    Visualize ground truth and predicted trajectories
    """
    # Convert to numpy arrays if they are tensors
    if torch.is_tensor(gt_action):
        gt_action = gt_action.cpu().numpy()
    if torch.is_tensor(pred_action):
        pred_action = pred_action.cpu().numpy()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each action dimension
    n_actions = gt_action.shape[-1]
    for i in range(n_actions):
        ax.plot(gt_action[:, i], label=f'GT Action {i+1}', linestyle='-', alpha=0.8)
        ax.plot(pred_action[:, i], label=f'Pred Action {i+1}', linestyle='--', alpha=0.8)
    
    ax.set_title(title)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action Value')
    ax.legend()
    ax.grid(True)
    
    return fig

def demo_hierarchical_diffusion():
    """
    Demo script for hierarchical diffusion policy with adversarial interaction
    """
    # Load configuration
    config_path = pathlib.Path(__file__).parent.parent / 'config'
    config_name = 'train_diffusion_hierarchical_workspace'
    
    with hydra.initialize_config_dir(config_dir=str(config_path)):
        cfg = hydra.compose(config_name=config_name)
    
    print("=== Hierarchical Diffusion Policy Demo ===")
    print("Configuration loaded successfully!")
    print(f"Latent dimensions: z_nav={cfg.latent_dim_nav}, z_adv={cfg.latent_dim_adv}")
    print(f"KL weights: nav={cfg.kl_weight}, adv={cfg.kl_weight_adv}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print("\n=== Initializing Model ===")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.to(device)
    policy.eval()
    
    print("Model initialized successfully!")
    print(f"Policy type: {type(policy).__name__}")
    print(f"Encoder type: {type(policy.vit_encoder).__name__}")
    
    # Create dummy data for testing
    print("\n=== Creating Dummy Data ===")
    batch_size = 2
    
    # Dummy observation and goal images
    context_size = cfg.vit_encoder.context_size
    obs_img = torch.randn(batch_size, 3*context_size, 128, 128, device=device)
    goal_img = torch.randn(batch_size, 3, 128, 128, device=device)
    
    # Dummy low-dimensional observations
    obs = torch.randn(batch_size, cfg.horizon, cfg.obs_dim, device=device)
    
    # Dummy actions
    action = torch.randn(batch_size, cfg.horizon, cfg.action_dim, device=device)
    
    # Create different adversarial masks
    adv_mask_disabled = torch.zeros(batch_size, device=device)  # Disable adversarial behavior
    adv_mask_enabled = torch.ones(batch_size, device=device)   # Enable adversarial behavior
    
    # Test 1: Generate trajectory with adversarial behavior disabled (mask=0)
    print("\n=== Test 1: Generating Trajectory (Adversarial Behavior Disabled) ===")
    obs_dict_disabled = {
        'obs': obs,
        'obs_img': obs_img,
        'goal_img': goal_img,
        'adv_mask': adv_mask_disabled
    }
    
    with torch.no_grad():
        result_disabled = policy.predict_action(obs_dict_disabled)
    
    print("Trajectory generated successfully with adversarial behavior disabled!")
    print(f"Predicted action shape: {result_disabled['action'].shape}")
    print(f"Latent z shape: {result_disabled['z'].shape}")
    print(f"Navigation intent (z_nav) shape: {result_disabled['z_nav_mean'].shape}")
    print(f"Adversarial intent (z_adv) shape: {result_disabled['z_adv_mean'].shape}")
    
    # Test 2: Generate trajectory with adversarial behavior enabled (mask=1)
    print("\n=== Test 2: Generating Trajectory (Adversarial Behavior Enabled) ===")
    obs_dict_enabled = {
        'obs': obs,
        'obs_img': obs_img,
        'goal_img': goal_img,
        'adv_mask': adv_mask_enabled
    }
    
    with torch.no_grad():
        result_enabled = policy.predict_action(obs_dict_enabled)
    
    print("Trajectory generated successfully with adversarial behavior enabled!")
    print(f"Predicted action shape: {result_enabled['action'].shape}")
    
    # Test 3: Optimize adversarial intent during inference
    print("\n=== Test 3: Optimizing Adversarial Intent During Inference ===")
    with torch.no_grad():
        result_optimized = policy.optimize_adversarial_intent(obs_dict_enabled, num_steps=5, lr=0.01)
    
    print("Adversarial intent optimization completed!")
    print(f"Optimized action shape: {result_optimized['action'].shape}")
    print(f"Original z_adv shape: {result_optimized['z_adv'].shape}")
    print(f"Optimized z_adv shape: {result_optimized['z_adv_opt'].shape}")
    
    # Test 4: Compute loss (training mode)
    print("\n=== Test 4: Computing Loss (Training Mode) ===")
    policy.train()
    
    batch = {
        'obs': obs,
        'obs_img': obs_img,
        'goal_img': goal_img,
        'action': action,
        'adv_mask': adv_mask_enabled
    }
    
    loss_dict = policy.compute_loss(batch)
    
    print("Loss computation completed!")
    print(f"Total loss: {loss_dict['loss'].item():.6f}")
    print(f"Diffusion loss: {loss_dict['diffusion_loss'].item():.6f}")
    print(f"Navigation KL loss: {loss_dict['kl_loss_nav'].item():.6f}")
    print(f"Adversarial KL loss: {loss_dict['kl_loss_adv'].item():.6f}")
    print(f"Total KL loss: {loss_dict['kl_loss_total'].item():.6f}")
    
    # Test 5: Compare trajectories with/without adversarial behavior
    print("\n=== Test 5: Comparing Trajectories ===")
    
    # For demonstration purposes, let's create ground truth actions
    gt_action = action[0].detach().cpu().numpy()
    
    # Get predicted actions
    pred_action_disabled = result_disabled['action'][0].detach().cpu().numpy()
    pred_action_enabled = result_enabled['action'][0].detach().cpu().numpy()
    pred_action_optimized = result_optimized['action'][0].detach().cpu().numpy()
    
    # Calculate MSE errors
    mse_disabled = np.mean((gt_action - pred_action_disabled) ** 2)
    mse_enabled = np.mean((gt_action - pred_action_enabled) ** 2)
    mse_optimized = np.mean((gt_action - pred_action_optimized) ** 2)
    
    print(f"MSE (Disabled Adversarial): {mse_disabled:.6f}")
    print(f"MSE (Enabled Adversarial): {mse_enabled:.6f}")
    print(f"MSE (Optimized Adversarial): {mse_optimized:.6f}")
    
    # Visualize trajectories (optional, comment out if not needed)
    print("\n=== Visualizing Trajectories ===")
    try:
        fig1 = visualize_trajectory(gt_action, pred_action_disabled, "Trajectory Comparison (Adversarial Disabled)")
        fig2 = visualize_trajectory(gt_action, pred_action_enabled, "Trajectory Comparison (Adversarial Enabled)")
        fig3 = visualize_trajectory(gt_action, pred_action_optimized, "Trajectory Comparison (Adversarial Optimized)")
        
        # Save figures
        output_dir = pathlib.Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        fig1.savefig(output_dir / "trajectory_disabled.png")
        fig2.savefig(output_dir / "trajectory_enabled.png")
        fig3.savefig(output_dir / "trajectory_optimized.png")
        
        print(f"Trajectory visualizations saved to {output_dir}")
        plt.close('all')
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("\n=== Demo Complete ===")
    print("All tests passed successfully!")
    print("\nKey features demonstrated:")
    print("1. Hierarchical latent space generation (z_nav, z_adv)")
    print("2. Binary mask control for adversarial behavior")
    print("3. Trajectory generation with diffusion policy")
    print("4. Inference-time adversarial intent optimization")
    print("5. Mask-aware loss computation")
    
    return {
        'result_disabled': result_disabled,
        'result_enabled': result_enabled,
        'result_optimized': result_optimized,
        'loss_dict': loss_dict
    }

if __name__ == "__main__":
    demo_hierarchical_diffusion()