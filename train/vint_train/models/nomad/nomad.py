# import os
# import argparse
# import time
# import pdb

# import torch
# import torch.nn as nn


# class NoMaD(nn.Module):

#     def __init__(self, vision_encoder, 
#                        noise_pred_net,
#                        dist_pred_net,
#                        z_dim=16):   # 新增 latent z 维度
#         super(NoMaD, self).__init__()

#         self.vision_encoder = vision_encoder
#         self.noise_pred_net = noise_pred_net
#         self.dist_pred_net = dist_pred_net
#         self.z_dim = z_dim
    
#     def forward(self, func_name, **kwargs):
#         """
#         func_name: 指定调用的子网络
#         kwargs:
#             latent_z: latent 条件向量，可选
#             lambda_cfg: CFG 放大系数，可选
#             其他参数视子网络而定
#         """
#         if func_name == "vision_encoder":
#             output = self.vision_encoder(
#                 kwargs["obs_img"], 
#                 kwargs["goal_img"], 
#                 input_goal_mask=kwargs.get("input_goal_mask", None)
#             )
        
#         elif func_name == "noise_pred_net":
#             # 提取 latent z
#             latent_z = kwargs.get("latent_z", None)  # shape: [batch, z_dim]
#             lambda_cfg = kwargs.get("lambda_cfg", 1.0)

#             # 如果使用 CFG，则生成 uncond 与 cond 两个预测
#             if latent_z is not None and lambda_cfg > 0.0:
#                 eps_cond = self.noise_pred_net(
#                     sample=kwargs["sample"], 
#                     timestep=kwargs["timestep"], 
#                     global_cond=kwargs["global_cond"], 
#                     latent_cond=latent_z
#                 )
#                 eps_uncond = self.noise_pred_net(
#                     sample=kwargs["sample"], 
#                     timestep=kwargs["timestep"], 
#                     global_cond=kwargs["global_cond"], 
#                     latent_cond=None
#                 )
#                 # CFG 修正
#                 output = eps_uncond + lambda_cfg * (eps_cond - eps_uncond)
#             else:
#                 output = self.noise_pred_net(
#                     sample=kwargs["sample"], 
#                     timestep=kwargs["timestep"], 
#                     global_cond=kwargs["global_cond"], 
#                     latent_cond=latent_z
#                 )

#         elif func_name == "dist_pred_net":
#             # 对 dist_pred_net 也可以加入 latent 支持，如果需要
#             latent_z = kwargs.get("latent_z", None)
#             output = self.dist_pred_net(
#                 obsgoal_cond=kwargs["obsgoal_cond"], 
#                 latent_cond=latent_z
#             )
        
#         else:
#             raise NotImplementedError

#         return output


# class DenseNetwork(nn.Module):
#     def __init__(self, embedding_dim):
#         super(DenseNetwork, self).__init__()
        
#         self.embedding_dim = embedding_dim 
#         self.network = nn.Sequential(
#             nn.Linear(self.embedding_dim, self.embedding_dim//4),
#             nn.ReLU(),
#             nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
#             nn.ReLU(),
#             nn.Linear(self.embedding_dim//16, 1)
#         )
    
#     def forward(self, x):
#         x = x.reshape((-1, self.embedding_dim))
#         output = self.network(x)
#         return output

import torch
import torch.nn as nn


class NoMaD(nn.Module):

    def __init__(
        self,
        vision_encoder,
        noise_pred_net,
        dist_pred_net,
        z_dim=16,
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net

        self.z_dim = z_dim

        # latent → condition 的投影（非常关键）
        self.z_proj = nn.Linear(z_dim, noise_pred_net.global_cond_dim)

    def _merge_cond(self, global_cond, latent_z):
        """
        将 latent z 融合进 global_cond
        global_cond: [B, C]
        latent_z:   [B, z_dim]
        """
        if latent_z is None:
            return global_cond

        z_emb = self.z_proj(latent_z)
        return global_cond + z_emb   # add 比 concat 更稳定

    def forward(self, func_name, **kwargs):

        if func_name == "vision_encoder":
            return self.vision_encoder(
                kwargs["obs_img"],
                kwargs["goal_img"],
                input_goal_mask=kwargs.get("input_goal_mask", None),
            )

        elif func_name == "noise_pred_net":

            latent_z = kwargs.get("latent_z", None)
            lambda_cfg = kwargs.get("lambda_cfg", 0.0)

            # ===== cond =====
            cond = self._merge_cond(
                kwargs["global_cond"], latent_z
            )

            eps_cond = self.noise_pred_net(
                sample=kwargs["sample"],
                timestep=kwargs["timestep"],
                global_cond=cond,
            )

            # ===== CFG =====
            if lambda_cfg > 0.0:
                # uncond = goal-masked condition
                uncond = self._merge_cond(
                    kwargs["global_cond_uncond"], latent_z
                )
                eps_uncond = self.noise_pred_net(
                    sample=kwargs["sample"],
                    timestep=kwargs["timestep"],
                    global_cond=uncond,
                )

                return eps_uncond + lambda_cfg * (eps_cond - eps_uncond)

            return eps_cond

        elif func_name == "dist_pred_net":
            return self.dist_pred_net(kwargs["obsgoal_cond"])

        else:
            raise NotImplementedError


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(embedding_dim // 16, 1),
        )

    def forward(self, x):
        return self.network(x.reshape(x.shape[0], -1))
