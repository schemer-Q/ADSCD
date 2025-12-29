import torch
from torch import nn
from einops import rearrange
from typing import Optional, Tuple
from train.vint_train.models.vint.vit import MaskedGoalViT

class ViTMLPEncoder(nn.Module):
    def __init__(
        self,
        obs_encoding_size: int = 512,
        context_size: int = 5,
        image_size: int = 128,
        patch_size: int = 16,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        latent_dim: int = 64,
    ) -> None:
        """
        ViT+MLP Encoder class that generates latent z distribution
        """
        super(ViTMLPEncoder, self).__init__()
        self.context_size = context_size
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        
        if type(image_size) == int:
            self.image_height = image_size
            self.image_width = image_size
        else:
            self.image_width = image_size[0]
            self.image_height = image_size[1]
        
        # Vision Transformer for visual encoding
        self.vit = MaskedGoalViT(
            context_size=context_size,
            image_size=(self.image_height, self.image_width*(self.context_size + 2)),
            patch_size=self.patch_size,
            dim=obs_encoding_size,
            depth=mha_num_attention_layers,
            heads=mha_num_attention_heads,
            mlp_dim=obs_encoding_size
        )
        
        # MLP for generating latent z distribution (mean and logvar)
        self.z_mean_mlp = nn.Sequential(
            nn.Linear(obs_encoding_size, obs_encoding_size // 2),
            nn.ReLU(),
            nn.Linear(obs_encoding_size // 2, latent_dim)
        )
        
        self.z_logvar_mlp = nn.Sequential(
            nn.Linear(obs_encoding_size, obs_encoding_size // 2),
            nn.ReLU(),
            nn.Linear(obs_encoding_size // 2, latent_dim)
        )
    
    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ViT+MLP Encoder
        
        Args:
            obs_img: Current and historical observations [B, C*context_size, H, W]
            goal_img: Target image [B, C, H, W]
            input_goal_mask: Optional goal mask [B]
            
        Returns:
            z_mean: Mean of latent distribution [B, latent_dim]
            z_logvar: Log variance of latent distribution [B, latent_dim]
        """
        # Process input images
        obs_img_list = list(torch.split(obs_img, 3, dim=1))
        obsgoal_img_list = obs_img_list + [goal_img]
        x = torch.cat(obsgoal_img_list, dim=-1)
        
        # Get ViT encoding
        vit_encoding = self.vit(x, input_goal_mask)
        
        # Generate latent distribution
        z_mean = self.z_mean_mlp(vit_encoding)
        z_logvar = self.z_logvar_mlp(vit_encoding)
        
        return z_mean, z_logvar
    
    def sample(self, z_mean: torch.Tensor, z_logvar: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample from latent distribution using reparameterization trick
        
        Args:
            z_mean: Mean of latent distribution [B, latent_dim]
            z_logvar: Log variance of latent distribution [B, latent_dim]
            deterministic: Whether to sample deterministically (use mean only)
            
        Returns:
            z: Sampled latent variable [B, latent_dim]
        """
        if deterministic:
            return z_mean
        
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std