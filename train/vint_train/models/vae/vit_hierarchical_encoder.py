import torch
from torch import nn
from einops import rearrange
from typing import Optional, Tuple
# Robust import for MaskedGoalViT: try normal import first; if it fails
# (e.g., running script from repo root where 'train' package isn't on sys.path),
# add project root and train dir to sys.path and retry.
# try:
    # Robust import for MaskedGoalViT: try normal import first; if it fails
# (e.g., running script from repo root where 'train' package isn't on sys.path),
# add project root and train dir to sys.path and retry.
try:
    from train.vint_train.models.vint.vit import MaskedGoalViT
except Exception:
    import sys
    import pathlib
    import importlib.util
    this_file = pathlib.Path(__file__).resolve()
    # ADSCD project root is three levels up from this file
    project_root = this_file.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Try to locate the vit file directly to avoid shadowing by files named
    # 'train.py' on sys.path. Build the expected file path and import it as a
    # module via importlib if present.
    vit_path = project_root / 'train' / 'vint_train' / 'models' / 'vint' / 'vit.py'
    if vit_path.exists():
        spec = importlib.util.spec_from_file_location("vit", str(vit_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MaskedGoalViT = module.MaskedGoalViT
    else:
        # re-raise original error if file not found
        raise
except Exception:
    import sys
    import pathlib
    import importlib.util
    this_file = pathlib.Path(__file__).resolve()
    # ADSCD project root is three levels up from this file
    project_root = this_file.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Try to locate the vit file directly to avoid shadowing by files named
    # 'train.py' on sys.path. Build the expected file path and import it as a
    # module via importlib if present.
    vit_path = project_root / 'train' / 'vint_train' / 'models' / 'vint' / 'vit.py'
    if vit_path.exists():
        spec = importlib.util.spec_from_file_location("vit", str(vit_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MaskedGoalViT = module.MaskedGoalViT
    else:
        # re-raise original error if file not found
        raise

class ViTHierarchicalEncoder(nn.Module):
    def __init__(
        self,
        obs_encoding_size: int = 512,
        context_size: int = 5,
        image_size: int = 128,
        patch_size: int = 16,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        latent_dim_nav: int = 64,
        latent_dim_adv: int = 32,
    ) -> None:
        """
        ViT+MLP Hierarchical Encoder class that generates hierarchical latent z distribution
        z = [z_nav, z_adv], where z_nav is navigation intent and z_adv is adversarial intent
        """
        super(ViTHierarchicalEncoder, self).__init__()
        self.context_size = context_size
        self.patch_size = patch_size
        self.latent_dim_nav = latent_dim_nav
        self.latent_dim_adv = latent_dim_adv
        self.latent_dim = latent_dim_nav + latent_dim_adv
        
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
        
        # MLP for generating navigation intent latent z_nav distribution
        self.z_nav_mean_mlp = nn.Sequential(
            nn.Linear(obs_encoding_size, obs_encoding_size // 2),
            nn.ReLU(),
            nn.Linear(obs_encoding_size // 2, latent_dim_nav)
        )
        
        self.z_nav_logvar_mlp = nn.Sequential(
            nn.Linear(obs_encoding_size, obs_encoding_size // 2),
            nn.ReLU(),
            nn.Linear(obs_encoding_size // 2, latent_dim_nav)
        )
        
        # MLP for generating adversarial intent latent z_adv distribution
        self.z_adv_mean_mlp = nn.Sequential(
            nn.Linear(obs_encoding_size, obs_encoding_size // 2),
            nn.ReLU(),
            nn.Linear(obs_encoding_size // 2, latent_dim_adv)
        )
        
        self.z_adv_logvar_mlp = nn.Sequential(
            nn.Linear(obs_encoding_size, obs_encoding_size // 2),
            nn.ReLU(),
            nn.Linear(obs_encoding_size // 2, latent_dim_adv)
        )
    
    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor, 
        adv_mask: Optional[torch.tensor] = None,
        input_goal_mask: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of ViT+MLP Hierarchical Encoder
        
        Args:
            obs_img: Current and historical observations [B, C*context_size, H, W]
            goal_img: Target image [B, C, H, W]
            adv_mask: Binary mask controlling adversarial intent activation [B, 1]
                     0: disable adversarial intent, 1: enable adversarial intent
            input_goal_mask: Optional goal mask [B]
            
        Returns:
            z_nav_mean: Mean of navigation latent distribution [B, latent_dim_nav]
            z_nav_logvar: Log variance of navigation latent distribution [B, latent_dim_nav]
            z_adv_mean: Mean of adversarial latent distribution [B, latent_dim_adv]
            z_adv_logvar: Log variance of adversarial latent distribution [B, latent_dim_adv]
        """
        # Process input images
        obs_img_list = list(torch.split(obs_img, 3, dim=1))
        obsgoal_img_list = obs_img_list + [goal_img]
        x = torch.cat(obsgoal_img_list, dim=-1)
        
        # Get ViT encoding
        vit_encoding = self.vit(x, input_goal_mask)
        
        # Generate navigation intent latent distribution
        z_nav_mean = self.z_nav_mean_mlp(vit_encoding)
        z_nav_logvar = self.z_nav_logvar_mlp(vit_encoding)
        
        # Generate adversarial intent latent distribution
        z_adv_mean = self.z_adv_mean_mlp(vit_encoding)
        z_adv_logvar = self.z_adv_logvar_mlp(vit_encoding)
        
        # Apply mask to adversarial intent (if provided)
        if adv_mask is not None:
            # Ensure mask is properly shaped [B, 1]
            adv_mask = adv_mask.view(-1, 1)
            # Zero out adversarial intent when mask is 0
            z_adv_mean = z_adv_mean * adv_mask
            z_adv_logvar = z_adv_logvar * adv_mask - 1e10 * (1 - adv_mask)  # Make logvar very small when masked
        
        return z_nav_mean, z_nav_logvar, z_adv_mean, z_adv_logvar
    
    def sample(self, z_nav_mean: torch.Tensor, z_nav_logvar: torch.Tensor, 
               z_adv_mean: torch.Tensor, z_adv_logvar: torch.Tensor, 
               adv_mask: Optional[torch.tensor] = None,
               deterministic: bool = False) -> torch.Tensor:
        """
        Sample from hierarchical latent distribution using reparameterization trick
        
        Args:
            z_nav_mean: Mean of navigation latent distribution [B, latent_dim_nav]
            z_nav_logvar: Log variance of navigation latent distribution [B, latent_dim_nav]
            z_adv_mean: Mean of adversarial latent distribution [B, latent_dim_adv]
            z_adv_logvar: Log variance of adversarial latent distribution [B, latent_dim_adv]
            adv_mask: Binary mask controlling adversarial intent activation [B, 1]
            deterministic: Whether to sample deterministically (use mean only)
            
        Returns:
            z: Sampled hierarchical latent variable [B, latent_dim_nav + latent_dim_adv]
        """
        if deterministic:
            z_nav = z_nav_mean
            z_adv = z_adv_mean
        else:
            # Sample navigation intent
            std_nav = torch.exp(0.5 * z_nav_logvar)
            eps_nav = torch.randn_like(std_nav)
            z_nav = z_nav_mean + eps_nav * std_nav
            
            # Sample adversarial intent
            std_adv = torch.exp(0.5 * z_adv_logvar)
            eps_adv = torch.randn_like(std_adv)
            z_adv = z_adv_mean + eps_adv * std_adv
        
        # Apply mask to adversarial intent (if provided)
        if adv_mask is not None:
            adv_mask = adv_mask.view(-1, 1)
            z_adv = z_adv * adv_mask
        
        # Concatenate navigation and adversarial intent
        z = torch.cat([z_nav, z_adv], dim=-1)
        
        return z
    
    def get_nav_mean_logvar(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract navigation intent from hierarchical latent variable
        """
        return z[:, :self.latent_dim_nav], torch.zeros_like(z[:, :self.latent_dim_nav])
    
    def get_adv_mean_logvar(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract adversarial intent from hierarchical latent variable
        """
        return z[:, self.latent_dim_nav:], torch.zeros_like(z[:, self.latent_dim_nav:])