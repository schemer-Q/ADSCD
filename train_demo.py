import sys
import os
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm

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

class SyntheticDataset(Dataset):
    def __init__(self, size=100, horizon=16, obs_dim=20, action_dim=2, other_state_dim=2, context_size=5):
        self.size = size
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.other_state_dim = other_state_dim
        self.context_size = context_size
        
        # Generate synthetic data
        # We create a simple rule: action = (goal - current) + g * (current - other)
        # To make it learnable from low-dim, we ensure low-dim obs contains current pos
        
        self.data = []
        for _ in range(size):
            # 1. State definitions
            # Assume obs[0:2] is current position (x, y)
            current_pos = torch.randn(2)
            goal_pos = torch.randn(2)
            other_pos = torch.randn(2)
            
            # 2. Create observation sequence (simplified: static for horizon)
            # obs: [horizon, obs_dim]
            obs = torch.randn(horizon, obs_dim)
            obs[:, :2] = current_pos # Embed position in first 2 dims
            
            # 3. Create other state
            other_state = other_pos # [other_state_dim] (assuming dim=2)
            
            # 4. Create mask (50% chance of interaction)
            g = torch.randint(0, 2, (1,)).float()
            
            # 5. Generate Ground Truth Action
            # Navigation component: move towards goal
            nav_vec = goal_pos - current_pos
            
            # Adversarial component: move away from other (if g=1)
            adv_vec = (current_pos - other_pos) * g.item()
            
            # Combined action
            target_action_step = nav_vec + adv_vec
            
            # Action sequence: [horizon, action_dim]
            # Add some noise to make it a trajectory
            action = target_action_step.unsqueeze(0).repeat(horizon, 1)
            action = action + torch.randn_like(action) * 0.1
            
            # 6. Images (Random noise for now, as we focus on logic flow)
            # In a real scenario, these would be rendered images corresponding to positions
            obs_img = torch.randn(3 * context_size, 128, 128)
            goal_img = torch.randn(3, 128, 128)
            
            self.data.append({
                'obs': obs,
                'obs_img': obs_img,
                'goal_img': goal_img,
                'action': action,
                'adv_mask': g,
                'other_state': other_state
            })

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

def train_demo():
    print("=== Adversarial Intent Factorization: Training Demo ===")
    
    # 1. Load Configuration
    config_path = project_root / 'diffusion_policy' / 'diffusion_policy' / 'config'
    config_name = 'train_diffusion_hierarchical_workspace'
    
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = hydra.compose(config_name=config_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Initialize Model
    print("\nInitializing Model...")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.to(device)
    
    # Initialize normalizer
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    
    # 3. Create Dataset and DataLoader
    print("\nGenerating Synthetic Dataset...")
    dataset_size = 100
    batch_size = 16
    dataset = SyntheticDataset(
        size=dataset_size,
        horizon=cfg.horizon,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        other_state_dim=cfg.other_state_dim,
        context_size=cfg.vit_encoder.context_size
    )
    
    # Create a dummy batch for normalizer fitting
    dummy_data = {
        'obs': torch.stack([d['obs'] for d in dataset.data]),
        'action': torch.stack([d['action'] for d in dataset.data])
    }
    normalizer.fit(dummy_data)
    policy.set_normalizer(normalizer)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset created. Size: {dataset_size}, Batch size: {batch_size}")
    
    # 4. Optimizer
    optimizer = optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-6)
    
    # 5. Training Loop
    num_epochs = 5
    print(f"\nStarting Training for {num_epochs} epochs...")
    
    policy.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_diff_loss = 0
        total_kl_nav = 0
        total_kl_adv = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass & Loss computation
            loss_dict = policy.compute_loss(batch)
            loss = loss_dict['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_diff_loss += loss_dict['diffusion_loss'].item()
            total_kl_nav += loss_dict['kl_loss_nav'].item()
            total_kl_adv += loss_dict['kl_loss_adv'].item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        avg_diff = total_diff_loss / len(dataloader)
        avg_kl_nav = total_kl_nav / len(dataloader)
        avg_kl_adv = total_kl_adv / len(dataloader)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Total Loss: {avg_loss:.6f}")
        print(f"  Diff Loss:  {avg_diff:.6f}")
        print(f"  KL Nav:     {avg_kl_nav:.6f}")
        print(f"  KL Adv:     {avg_kl_adv:.6f}")

    print("\nTraining Demo Completed.")

if __name__ == "__main__":
    train_demo()
