from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

# Robust import for ViTHierarchicalEncoder: try normal import first; if it fails
# (e.g., running script from repo root where 'train' package isn't on sys.path),
# add project root and train dir to sys.path and retry.
try:
    from train.vint_train.models.vae.vit_hierarchical_encoder import ViTHierarchicalEncoder
except Exception:
    import sys
    import pathlib
    this_file = pathlib.Path(__file__).resolve()
    # ADSCD project root is three levels up from this file
    project_root = this_file.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from train.vint_train.models.vae.vit_hierarchical_encoder import ViTHierarchicalEncoder

class DiffusionHierarchicalPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            vit_encoder: ViTHierarchicalEncoder,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            kl_weight=1.0,
            kl_weight_adv=1.0,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.vit_encoder = vit_encoder
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kl_weight = kl_weight
        self.kl_weight_adv = kl_weight_adv
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key, "goal_img" key, and optional "adv_mask" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'goal_img' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        
        # Extract inputs
        obs_img = obs_dict['obs_img']
        goal_img = obs_dict['goal_img']
        adv_mask = obs_dict.get('adv_mask', None)
        
        # Generate hierarchical latent z from visual inputs
        z_nav_mean, z_nav_logvar, z_adv_mean, z_adv_logvar = self.vit_encoder(obs_img, goal_img, adv_mask)
        z = self.vit_encoder.sample(z_nav_mean, z_nav_logvar, z_adv_mean, z_adv_logvar, adv_mask)
        
        # Normalize low-dimensional observations if provided
        nobs = None
        if 'obs' in obs_dict:
            nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        
        B = obs_img.shape[0]
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = z  # Use hierarchical latent z as global condition
        
        if 'obs' in obs_dict:
            nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
            if self.obs_as_global_cond:
                # Concatenate latent z with flattened observations
                obs_flat = nobs[:,:self.n_obs_steps,:].reshape(nobs.shape[0], -1)
                global_cond = torch.cat([z, obs_flat], dim=-1)

        # Prepare condition data and mask
        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_obs_steps
            if self.oa_step_convention:
                start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'z': z,
            'z_nav_mean': z_nav_mean,
            'z_nav_logvar': z_nav_logvar,
            'z_adv_mean': z_adv_mean,
            'z_adv_logvar': z_adv_logvar
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        try:
            nbatch = self.normalizer.normalize(batch)
        except Exception:
            # Normalizer not initialized for these keys; fall back to raw batch
            # This allows demo/training to proceed when no fitted normalizer is provided.
            nbatch = batch
        
        # Extract inputs
        obs_img = batch['obs_img']
        goal_img = batch['goal_img']
        adv_mask = batch.get('adv_mask', None)
        
        # Generate hierarchical latent distribution from visual inputs
        z_nav_mean, z_nav_logvar, z_adv_mean, z_adv_logvar = self.vit_encoder(obs_img, goal_img, adv_mask)
        z = self.vit_encoder.sample(z_nav_mean, z_nav_logvar, z_adv_mean, z_adv_logvar, adv_mask)
        
        # Compute KL divergence losses with condition mask
        # Navigation intent KL loss (always active)
        kl_loss_nav = -0.5 * torch.sum(1 + z_nav_logvar - z_nav_mean.pow(2) - z_nav_logvar.exp(), dim=1)
        
        # Adversarial intent KL loss (mask-aware)
        kl_loss_adv = -0.5 * torch.sum(1 + z_adv_logvar - z_adv_mean.pow(2) - z_adv_logvar.exp(), dim=1)
        if adv_mask is not None:
            # Apply mask to adversarial KL loss
            adv_mask = adv_mask.view(-1)
            kl_loss_adv = kl_loss_adv * adv_mask
        
        # Mean over batch
        kl_loss_nav = kl_loss_nav.mean()
        kl_loss_adv = kl_loss_adv.mean()
        
        # Prepare diffusion model inputs
        obs = nbatch['obs'] if 'obs' in nbatch else None
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = z  # Use hierarchical latent z as global condition
        trajectory = action
        
        if obs is not None:
            if self.obs_as_local_cond:
                # zero out observations after n_obs_steps
                local_cond = obs
                local_cond[:,self.n_obs_steps:,:] = 0
            elif self.obs_as_global_cond:
                # Concatenate latent z with flattened observations
                obs_flat = obs[:,:self.n_obs_steps,:].reshape(obs.shape[0], -1)
                global_cond = torch.cat([z, obs_flat], dim=-1)
                
                if self.pred_action_steps_only:
                    To = self.n_obs_steps
                    start = To
                    if self.oa_step_convention:
                        start = To - 1
                    end = start + self.n_action_steps
                    trajectory = action[:,start:end]
            else:
                trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # Compute diffusion loss (mask-aware)
        diffusion_loss = F.mse_loss(pred, target, reduction='none')
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = diffusion_loss.mean()
        
        # Total loss: diffusion loss + KL losses
        total_loss = diffusion_loss + \
                    self.kl_weight * kl_loss_nav + \
                    self.kl_weight_adv * kl_loss_adv
        
        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'kl_loss_nav': kl_loss_nav,
            'kl_loss_adv': kl_loss_adv,
            'kl_loss_total': self.kl_weight * kl_loss_nav + self.kl_weight_adv * kl_loss_adv
        }
    
    def optimize_adversarial_intent(self, obs_dict: Dict[str, torch.Tensor], 
                                   num_steps: int = 10, lr: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Optimize adversarial intent z_adv to improve adversarial performance during inference
        
        Args:
            obs_dict: Observation dictionary containing obs_img, goal_img, and adv_mask
            num_steps: Number of optimization steps
            lr: Learning rate for optimization
            
        Returns:
            result: Dictionary containing optimized action and latent variables
        """
        # Set encoder to eval mode but allow gradients for z_adv
        self.vit_encoder.eval()
        self.model.eval()
        
        # Extract inputs
        obs_img = obs_dict['obs_img'].detach()
        goal_img = obs_dict['goal_img'].detach()
        adv_mask = obs_dict.get('adv_mask', None)
        
        # Generate navigation and adversarial intent distributions
        z_nav_mean, z_nav_logvar, z_adv_mean, z_adv_logvar = self.vit_encoder(obs_img, goal_img, adv_mask)

        # For deterministic optimization we use the distribution means directly
        z_nav = z_nav_mean.detach()  # Fix navigation intent

        # Initialize adversarial intent as learnable parameter using its mean
        z_adv = z_adv_mean.detach().requires_grad_(True)
        
        # Optimizer for z_adv
        optimizer = torch.optim.Adam([z_adv], lr=lr)
        
        # Normalize observations
        nobs = None
        if 'obs' in obs_dict:
            nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        
        B = obs_img.shape[0]
        T = self.horizon
        Da = self.action_dim
        
        device = self.device
        dtype = self.dtype
        
        def _compute_action_pred_from_z(z_tensor):
            """Helper: given full z (nav+adv), run conditional_sample and return unnormalized action_pred."""
            local_cond = None
            global_cond = z_tensor
            if 'obs' in obs_dict and self.obs_as_global_cond:
                obs_flat = nobs[:,:self.n_obs_steps,:].reshape(nobs.shape[0], -1)
                global_cond = torch.cat([z_tensor, obs_flat], dim=-1)

            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            nsample = self.conditional_sample(
                cond_data, cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs)

            naction_pred = nsample[...,:Da]
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
            return action_pred

        use_fd = False
        for step in range(num_steps):
            # Apply mask to adversarial intent
            if adv_mask is not None:
                z_adv_masked = z_adv * adv_mask.view(-1, 1)
            else:
                z_adv_masked = z_adv

            # Concatenate navigation and adversarial intent
            z = torch.cat([z_nav, z_adv_masked], dim=-1)

            # Compute action prediction for current z
            action_pred = _compute_action_pred_from_z(z)

            # Simple adversarial loss (replace with task-specific loss)
            adv_loss = -torch.mean(action_pred)

            # If adv_loss does not require grad (sampling path not differentiable),
            # fall back to finite-difference optimization
            if not adv_loss.requires_grad:
                use_fd = True
                break

            # Otherwise use gradient-based update
            optimizer.zero_grad()
            adv_loss.backward()
            optimizer.step()

        if use_fd:
            # Finite-difference numerical gradient optimization over z_adv
            eps = 1e-3
            for step in range(num_steps):
                grads = torch.zeros_like(z_adv)
                for i in range(z_adv.shape[1]):
                    z_plus = z_adv.clone()
                    z_minus = z_adv.clone()
                    z_plus[:, i] = z_plus[:, i] + eps
                    z_minus[:, i] = z_minus[:, i] - eps

                    if adv_mask is not None:
                        z_plus_masked = z_plus * adv_mask.view(-1,1)
                        z_minus_masked = z_minus * adv_mask.view(-1,1)
                    else:
                        z_plus_masked = z_plus
                        z_minus_masked = z_minus

                    z_full_plus = torch.cat([z_nav, z_plus_masked], dim=-1)
                    z_full_minus = torch.cat([z_nav, z_minus_masked], dim=-1)

                    action_plus = _compute_action_pred_from_z(z_full_plus)
                    action_minus = _compute_action_pred_from_z(z_full_minus)

                    loss_plus = -torch.mean(action_plus)
                    loss_minus = -torch.mean(action_minus)

                    # central difference
                    grad_i = (loss_plus - loss_minus) / (2 * eps)
                    # assign same gradient for all batch entries
                    grads[:, i] = grad_i.detach()

                # gradient ascent (we maximize adv objective here)
                with torch.no_grad():
                    z_adv += lr * grads
        
        # Get final optimized action
        with torch.no_grad():
            # Apply mask to adversarial intent
            if adv_mask is not None:
                z_adv_masked = z_adv * adv_mask.view(-1, 1)
            else:
                z_adv_masked = z_adv
            
            # Concatenate navigation and adversarial intent
            z = torch.cat([z_nav, z_adv_masked], dim=-1)
            
            # Prepare diffusion model inputs
            local_cond = None
            global_cond = z
            
            if 'obs' in obs_dict:
                if self.obs_as_global_cond:
                    obs_flat = nobs[:,:self.n_obs_steps,:].reshape(nobs.shape[0], -1)
                    global_cond = torch.cat([z, obs_flat], dim=-1)
            
            # Generate final action trajectory
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            
            # Sample trajectory
            nsample = self.conditional_sample(
                cond_data, cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs)
            
            # Unnormalize action
            naction_pred = nsample[...,:Da]
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
            
            # Get action
            if self.pred_action_steps_only:
                action = action_pred
            else:
                start = self.n_obs_steps
                if self.oa_step_convention:
                    start = self.n_obs_steps - 1
                end = start + self.n_action_steps
                action = action_pred[:,start:end]
        
        return {
            'action': action,
            'action_pred': action_pred,
            'z': z,
            'z_nav': z_nav,
            'z_adv': z_adv.detach(),
            'z_adv_opt': z_adv_masked.detach()
        }