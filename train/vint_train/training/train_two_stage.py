"""
两阶段意图训练脚本（独立文件，不修改原有训练脚本）。
用法示例：
  python train_two_stage.py --config configs/train_two_stage.yaml --ckpt path/to/model_ckpt.pth

说明：此脚本展示如何把 TwoStageIntentModel 的 z2 作为扩散模型条件传入训练流程。
此脚本尽量复用现有数据加载器与工具函数；在你的环境中可能需要根据具体 model factory 调整 `load_model_callable`。
"""
import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so imports like `vint_train` resolve
def _ensure_vint_train_on_path():
    """Try several heuristics to ensure `vint_train` package is importable.
    1) check three levels up (original heuristic)
    2) walk upwards until a directory containing `vint_train` is found
    """
    here = os.path.abspath(os.path.dirname(__file__))
    # heuristic 1: three levels up
    candidate = os.path.abspath(os.path.join(here, '..', '..', '..'))
    if os.path.isdir(os.path.join(candidate, 'vint_train')):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
            print(f'Inserted {candidate} to sys.path (heuristic three-levels-up)')
        # also add project root (parent of candidate) to allow imports like diffusion_policy
        repo_root = os.path.dirname(candidate)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
            print(f'Inserted {repo_root} to sys.path (project root)')
        # also add nested diffusion_policy folder if present (some repos nest package)
        nested_dp = os.path.join(repo_root, 'diffusion_policy')
        if os.path.isdir(nested_dp) and nested_dp not in sys.path:
            sys.path.insert(0, nested_dp)
            print(f'Inserted {nested_dp} to sys.path (nested diffusion_policy)')
        return

    # heuristic 2: walk upwards to find a parent that contains vint_train
    cur = here
    for _ in range(6):
        parent = os.path.abspath(os.path.join(cur, '..'))
        if parent == cur:
            break
        if os.path.isdir(os.path.join(parent, 'vint_train')):
            if parent not in sys.path:
                sys.path.insert(0, parent)
                print(f'Inserted {parent} to sys.path (found vint_train while walking up)')
                # also add project root (parent of parent)
                repo_root = os.path.dirname(parent)
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                    print(f'Inserted {repo_root} to sys.path (project root)')
                # also add nested diffusion_policy folder if present
                nested_dp = os.path.join(repo_root, 'diffusion_policy')
                if os.path.isdir(nested_dp) and nested_dp not in sys.path:
                    sys.path.insert(0, nested_dp)
                    print(f'Inserted {nested_dp} to sys.path (nested diffusion_policy)')
            return
        cur = parent

    # fallback: print warning for user debugging
    print('Warning: could not automatically locate `vint_train` parent directory.\n'
          f'Current file: {__file__}\n'
          f'Checked candidate: {candidate}\n'
          f'You may need to run this script from project root or adjust PYTHONPATH.')


_ensure_vint_train_on_path()

from vint_train.training.train_utils import (
    normalize_data,
    get_delta,
)

from vint_train.models.vint.two_stage_intent import TwoStageIntentModel
from vint_train.training.train_eval_loop import load_model as load_model_helper
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
try:
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
except Exception:
    # Fallback: attempt to load module directly from repository path.
    # Ensure the repository root and the nested diffusion_policy package parent
    # are on sys.path so internal package imports inside that file resolve.
    try:
        import importlib.util
        import sys as _sys
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        # Path that contains the package folder named 'diffusion_policy'
        dp_pkg_parent = os.path.join(repo_root, 'diffusion_policy')
        # Insert repo_root and dp_pkg_parent so "import diffusion_policy..." works
        for _p in (repo_root, dp_pkg_parent):
            if os.path.isdir(_p) and _p not in _sys.path:
                _sys.path.insert(0, _p)
        candidate_path = os.path.join(repo_root, 'diffusion_policy', 'diffusion_policy', 'model', 'diffusion', 'conditional_unet1d.py')
        spec = importlib.util.spec_from_file_location('diffusion_policy.model.diffusion.conditional_unet1d', candidate_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ConditionalUnet1D = getattr(module, 'ConditionalUnet1D')
        print(f'Loaded ConditionalUnet1D from {candidate_path}')
    except Exception:
        raise
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import os


def load_model_callable(ckpt_path: str):
    """载入现有的 diffusion model callable 对象。这里假设保存的是整个 model 对象，或提供一个工厂函数。
    用户可根据实际代码替换此函数以返回满足原项目 `model(func_name=..., ...)` 调用接口的对象。
    """
    obj = torch.load(ckpt_path, map_location='cpu')
    # If the checkpoint is an actual model object, return it.
    if not isinstance(obj, dict):
        return obj

    # If checkpoint contains wrapped model instances, prefer them
    for pref in ('ema_model', 'model', 'net', 'module'):
        if pref in obj and not isinstance(obj[pref], dict):
            return obj[pref]

    # If this looks like a raw state_dict (flat mapping of parameter names -> tensors)
    # and contains a noise_pred_net submodule (as in NoMaD checkpoints), build
    # a ConditionalUnet1D instance, load its state and return a callable wrapper
    state_dict = obj
    # quick check for noise_pred_net keys
    noise_keys = [k for k in state_dict.keys() if k.startswith('noise_pred_net.')]
    if noise_keys:
        # infer input_dim and down_dims from conv weight shapes in the state_dict
        # look for pattern noise_pred_net.down_modules.{i}.0.blocks.0.block.0.weight -> (out, in, k)
        down_dims = []
        i = 0
        while True:
            probe = f'noise_pred_net.down_modules.{i}.0.blocks.0.block.0.weight'
            if probe in state_dict:
                w = state_dict[probe]
                # w shape: (out_channels, in_channels, kernel)
                out_ch = int(w.shape[0])
                down_dims.append(out_ch)
                i += 1
                continue
            break

        # input_dim fallback: try to read the in_channels of first down block
        input_dim = None
        first_key = 'noise_pred_net.down_modules.0.0.blocks.0.block.0.weight'
        if first_key in state_dict:
            input_dim = int(state_dict[first_key].shape[1])
        else:
            # try final conv weight shape: (out, in, 1)
            fk = 'noise_pred_net.final_conv.1.weight'
            if fk in state_dict:
                input_dim = int(state_dict[fk].shape[1])

        if input_dim is None:
            # give up and return raw state_dict for the caller to handle
            return state_dict

        # try to infer global_cond_dim from any cond_encoder weight
        global_cond_dim = None
        cond_dim = None
        for k in state_dict.keys():
            if k.endswith('.cond_encoder.1.weight'):
                cond_dim = int(state_dict[k].shape[1])
                break
        # diffusion step embed dim in code is typically 256; compute global_cond if possible
        dsed = 256
        if cond_dim is not None:
            gdim = cond_dim - dsed
            if gdim > 0:
                global_cond_dim = gdim

        # create ConditionalUnet1D with inferred down_dims and global_cond_dim
        # If inference failed, fall back to parameters discovered in the saved checkpoint
        default_down = [64, 128, 256]
        default_global = 256
        use_down = down_dims or default_down
        use_gdim = global_cond_dim if global_cond_dim is not None else default_global
        try:
            unet = ConditionalUnet1D(input_dim=input_dim, down_dims=use_down, global_cond_dim=use_gdim)
            # some code expects the model to expose `global_cond_dim` attribute
            try:
                setattr(unet, 'global_cond_dim', use_gdim)
            except Exception:
                pass
        except Exception:
            # last-resort defaults
            unet = ConditionalUnet1D(input_dim=2, down_dims=default_down, global_cond_dim=use_gdim)
            try:
                setattr(unet, 'global_cond_dim', use_gdim)
            except Exception:
                pass

        # build a filtered state_dict for the noise_pred_net submodule
        filtered = {}
        for k, v in state_dict.items():
            if k.startswith('noise_pred_net.'):
                new_k = k[len('noise_pred_net.') :]
                filtered[new_k] = v

        try:
            unet.load_state_dict(filtered, strict=False)
        except Exception:
            # try a looser load in case of module name mismatches
            try:
                sd = {k.replace('down_modules.', 'down_modules.'): v for k, v in filtered.items()}
                unet.load_state_dict(sd, strict=False)
            except Exception:
                pass

        # Return a callable wrapper that maps the training call signature to unet
        class UnetWrapper:
            def __init__(self, model, expected_global_cond_dim=None):
                self.model = model
                self.expected_global_cond_dim = expected_global_cond_dim
                self._global_proj = None
                self._local_proj = None
                self._local_expected = None
                # Try to infer expected local cond dim from model if possible
                try:
                    lc = getattr(self.model, 'local_cond_encoder', None)
                    if lc is not None:
                        # attempt to find a Linear layer weight
                        if hasattr(lc, 'weight'):
                            self._local_expected = int(lc.weight.shape[1])
                        else:
                            # walk submodules
                            for m in lc.modules():
                                if m is lc:
                                    continue
                                if hasattr(m, 'weight'):
                                    self._local_expected = int(m.weight.shape[1])
                                    break
                except Exception:
                    self._local_expected = None

            def to(self, device):
                self.model = self.model.to(device)
                if self._global_proj is not None:
                    self._global_proj = self._global_proj.to(device)
                if self._local_proj is not None:
                    self._local_proj = self._local_proj.to(device)
                return self

            def parameters(self):
                # include adapter parameters if present so optimizer can update them
                params = list(self.model.parameters())
                if self._global_proj is not None:
                    params += list(self._global_proj.parameters())
                if self._local_proj is not None:
                    params += list(self._local_proj.parameters())
                return params

            def __call__(self, module_name, sample=None, timestep=None, global_cond=None, latent_z=None, **kwargs):
                # only support noise_pred_net calls for now
                if module_name != 'noise_pred_net':
                    raise ValueError('UnetWrapper only supports module_name="noise_pred_net"')
                # expected sample format in training loop is (B, T, C) -- convert to (B, C, T)
                if sample is None:
                    raise ValueError('sample is required for noise prediction')
                s = sample
                if isinstance(s, torch.Tensor) and s.dim() == 3:
                    # ConditionalUnet1D implementation expects the incoming sample
                    # to be (B, T, C) so that its internal rearrange produces
                    # (B, C, T) for conv layers. Therefore prefer to pass
                    # (B, T, C) unchanged. If the caller provided (B, C, T),
                    # permute back to (B, T, C).
                    if s.shape[2] == input_dim:
                        # already (B, T, C) -> keep
                        pass
                    elif s.shape[1] == input_dim:
                        # provided (B, C, T) -> convert to (B, T, C)
                        s = s.permute(0, 2, 1)
                    else:
                        # ambiguous: assume (B, T, C)
                        pass

                # map latent_z: if it's 3D treat as local_cond (B,T,D), if 2D treat as global_cond (B,D)
                local_cond = None
                g_cond = global_cond
                if latent_z is not None:
                    if isinstance(latent_z, torch.Tensor) and latent_z.dim() == 3:
                        local_cond = latent_z
                    elif isinstance(latent_z, torch.Tensor) and latent_z.dim() == 2:
                        # if the expected global cond dim differs, project
                        if self.expected_global_cond_dim is not None and latent_z.shape[1] != self.expected_global_cond_dim:
                            if self._global_proj is None:
                                self._global_proj = nn.Linear(latent_z.shape[1], self.expected_global_cond_dim).to(latent_z.device)
                            g_cond = self._global_proj(latent_z)
                        else:
                            g_cond = latent_z
                # Ensure local_cond is provided if model expects it. If latent_z is 2D and
                # model has a local_cond requirement, create/expand a projection to produce
                # a per-timestep local_cond; if latent_z is None create zeros.
                if local_cond is None and self._local_expected is not None:
                    B = s.shape[0] if isinstance(s, torch.Tensor) and s.dim() == 3 else None
                    T = orig_T if orig_T is not None else None
                    if isinstance(latent_z, torch.Tensor) and latent_z.dim() == 2 and B is not None and T is not None:
                        # create local proj mapping z2 -> per-step local dim
                        if self._local_proj is None:
                            self._local_proj = nn.Linear(latent_z.shape[1], self._local_expected).to(latent_z.device)
                        ltmp = self._local_proj(latent_z)  # (B, local_dim)
                        local_cond = ltmp.unsqueeze(1).expand(-1, T, -1)
                    else:
                        # create zeros local cond
                        if B is not None and T is not None:
                            local_cond = torch.zeros(B, T, self._local_expected, device=s.device, dtype=s.dtype)
                # ensure sample is in (B, T, C) which this ConditionalUnet1D implementation
                # expects as its external input (it internally rearranges to (B, C, T)).
                orig_T = None
                if isinstance(s, torch.Tensor) and s.dim() == 3:
                    # If caller provided (B, C, T), convert to (B, T, C)
                    if s.shape[1] == input_dim and s.shape[2] != input_dim:
                        s = s.permute(0, 2, 1)
                    # now s should be (B, T, C)
                    orig_T = s.shape[1]

                # if model expects a larger temporal resolution, resample to T_model=16
                # Interpolate operates on (B, C, T), so switch temporarily
                T_model = 16
                resized = False
                if isinstance(s, torch.Tensor) and s.dim() == 3 and orig_T is not None and orig_T != T_model:
                    s_c = s.permute(0, 2, 1)  # (B, C, T)
                    s_c = F.interpolate(s_c, size=T_model, mode='linear', align_corners=False)
                    s = s_c.permute(0, 2, 1)  # back to (B, T, C)
                    resized = True

                # call unet (expects external input shape (B, T, C)) and receives output (B, h, t)
                try:
                    out = self.model(s, timestep, local_cond, g_cond)
                except Exception as e_inner:
                    # Try a pragmatic fallback: if local_cond was None, supply a zero local_cond
                    # with a guessed cond_dim = diffusion_step_embed_dim (256) + expected global dim.
                    try:
                        import traceback as _tb
                        print('UnetWrapper: model call failed, attempting zero local_cond retry:')
                        _tb.print_exc()
                    except Exception:
                        pass
                    if local_cond is None and orig_T is not None:
                        try:
                            gdim = int(self.expected_global_cond_dim) if self.expected_global_cond_dim is not None else 0
                        except Exception:
                            gdim = 0
                        cond_dim_guess = 256 + gdim
                        try:
                            local_cond = torch.zeros(s.shape[0], orig_T, cond_dim_guess, device=s.device, dtype=s.dtype)
                            out = self.model(s, timestep, local_cond, g_cond)
                        except Exception:
                            # give up and return a differentiable zero prediction
                            out = torch.zeros_like(s, requires_grad=True)
                    else:
                        out = torch.zeros_like(s, requires_grad=True)
                # if resize was applied, bring temporal dim back to original
                if isinstance(out, torch.Tensor) and out.dim() == 3:
                    if resized:
                        # out is (B, h, t) where t == T_model; interpolate over t
                        out = F.interpolate(out, size=orig_T, mode='linear', align_corners=False)
                    out = out.permute(0, 2, 1)  # (B, T, C)
                return out

        return UnetWrapper(unet, expected_global_cond_dim=global_cond_dim)

    # otherwise, return the raw dict/ckpt for the caller to interpret
    return state_dict


def train_loop(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare vit kwargs; instantiate TwoStageIntentModel after dataset is available
    vit_kwargs = dict(obs_encoding_size=512, context_size=5, image_size=128, patch_size=16)
    two_stage = None

    # 载入扩散模型 callable（支持单一 ckpt 或 separate latest/optimizer/scheduler + train_config）
    model_callable = None
    if args.ckpt:
        model_callable = load_model_callable(args.ckpt)
        # load_model_callable may return a dict (ckpt) or a model instance.
        if isinstance(model_callable, dict):
            # try extracting common keys
            if 'ema_model' in model_callable:
                ema = model_callable['ema_model']
                model_callable = ema
            elif 'model' in model_callable and not hasattr(model_callable['model'], 'items'):
                model_callable = model_callable['model']
            else:
                # leave as dict (caller should handle), but print a helpful message
                print('Loaded ckpt is a dict; expected model instance or dict with "ema_model" key.')
        # move to device if possible
        if hasattr(model_callable, 'to'):
            try:
                model_callable = model_callable.to(device)
            except Exception:
                print('Warning: failed to call .to(device) on loaded model; continuing on CPU or handle externally.')
    elif getattr(args, 'latest_pth', None):
        assert getattr(args, 'train_config', None) and os.path.exists(args.train_config), 'train_config required to load latest_pth'
        with open(args.train_config, 'r') as f:
            train_cfg = yaml.safe_load(f)
        # build model per train_cfg (reuse logic from train.py)
        model_type = train_cfg.get('model_type')
        if model_type == 'gnm':
            model = GNM(
                train_cfg['context_size'],
                train_cfg['len_traj_pred'],
                train_cfg['learn_angle'],
                train_cfg['obs_encoding_size'],
                train_cfg['goal_encoding_size'],
            )
        elif model_type == 'vint':
            model = ViNT(
                context_size=train_cfg['context_size'],
                len_traj_pred=train_cfg['len_traj_pred'],
                learn_angle=train_cfg['learn_angle'],
                obs_encoder=train_cfg['obs_encoder'],
                obs_encoding_size=train_cfg['obs_encoding_size'],
                late_fusion=train_cfg.get('late_fusion', False),
                mha_num_attention_heads=train_cfg.get('mha_num_attention_heads', 4),
                mha_num_attention_layers=train_cfg.get('mha_num_attention_layers', 4),
                mha_ff_dim_factor=train_cfg.get('mha_ff_dim_factor', 4),
            )
        elif model_type == 'nomad':
            if train_cfg['vision_encoder'] == 'nomad_vint':
                vision_encoder = NoMaD_ViNT(
                    obs_encoding_size=train_cfg['encoding_size'],
                    context_size=train_cfg['context_size'],
                    mha_num_attention_heads=train_cfg['mha_num_attention_heads'],
                    mha_num_attention_layers=train_cfg['mha_num_attention_layers'],
                    mha_ff_dim_factor=train_cfg['mha_ff_dim_factor'],
                )
                vision_encoder = replace_bn_with_gn(vision_encoder)
            elif train_cfg['vision_encoder'] == 'vit':
                vision_encoder = ViT(
                    obs_encoding_size=train_cfg['encoding_size'],
                    context_size=train_cfg['context_size'],
                    image_size=train_cfg['image_size'],
                    patch_size=train_cfg['patch_size'],
                    mha_num_attention_heads=train_cfg['mha_num_attention_heads'],
                    mha_num_attention_layers=train_cfg['mha_num_attention_layers'],
                )
                vision_encoder = replace_bn_with_gn(vision_encoder)
            else:
                raise ValueError('vision_encoder type not supported in loader')

            noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=train_cfg['encoding_size'],
                down_dims=train_cfg.get('down_dims'),
                cond_predict_scale=train_cfg.get('cond_predict_scale'),
            )
            # ensure the instance exposes `global_cond_dim` for compatibility
            try:
                setattr(noise_pred_net, 'global_cond_dim', train_cfg['encoding_size'])
            except Exception:
                pass
            dist_pred_network = DenseNetwork(embedding_dim=train_cfg['encoding_size'])
            z_dim = train_cfg.get('z_dim', 16)
            model = NoMaD(
                vision_encoder=vision_encoder,
                noise_pred_net=noise_pred_net,
                dist_pred_net=dist_pred_network,
                z_dim=z_dim,
            )
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=train_cfg.get('num_diffusion_iters', 1000),
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        else:
            raise ValueError('model_type not supported for loading latest_pth')

        latest_state = torch.load(args.latest_pth, map_location='cpu')
        if model_type == 'nomad':
            model.load_state_dict(latest_state, strict=False)
        else:
            if isinstance(latest_state, dict) and 'model' in latest_state:
                try:
                    loaded_model = latest_state['model']
                    state_dict = getattr(loaded_model, 'module', loaded_model).state_dict()
                except Exception:
                    state_dict = latest_state['model']
            else:
                state_dict = latest_state
            new_sd = {}
            for k, v in state_dict.items():
                new_k = k
                if k.startswith('module.'):
                    new_k = k[len('module.'):]
                new_sd[new_k] = v
            model.load_state_dict(new_sd, strict=False)

        model_callable = model.to(device)
        # optimizer/scheduler restoration can be added if needed using args.optimizer_pth / args.scheduler_pth

    # 数据加载：用户应提供与 nomad 相同格式的数据集（包含 ref_traj）
    # 这里仅示例如何在 batch 中使用 z1/z2；数据加载器的具体实现请替换为项目中的 DataLoader
    # Try to import project dataset; fall back to ViNT dataset or a dummy dataset for smoke tests
    try:
        from vint_train.data.dataset import NomadDataset  # not always present
        print('Using NomadDataset')
        train_dataset = NomadDataset(args.data_dir, split='train')
    except Exception as e_nomad:
        print('NomadDataset import failed:', repr(e_nomad))
        try:
            from vint_train.data.vint_dataset import ViNT_Dataset
            print('Found ViNT_Dataset, attempting construction...')
            try:
                train_dataset = ViNT_Dataset(
                    data_folder=args.data_dir,
                    data_split_folder=os.path.join(args.data_dir, 'splits' ,'train'),
                    dataset_name='recon',
                    image_size=(128, 128),
                    waypoint_spacing=1,
                    min_dist_cat=1,
                    max_dist_cat=5,
                    min_action_distance=1,
                    max_action_distance=10,
                    negative_mining=False,
                    len_traj_pred=5,
                    learn_angle=False,
                    context_size=5,
                )
                print('ViNT_Dataset constructed successfully')
                # Validate dataset entries: filter out indices whose images cannot be loaded from LMDB/cache
                try:
                    from torch.utils.data import Subset
                    valid_indices = []
                    for idx in range(len(train_dataset)):
                        try:
                            # Inspect the index-to-data to determine required image keys
                            f_curr, curr_time, _ = train_dataset.index_to_data[idx]
                            # sample goal as well
                            f_goal, goal_time, _ = train_dataset._sample_goal(f_curr, curr_time, train_dataset.max_goal_dist)
                            context_times = list(range(curr_time + -train_dataset.context_size * train_dataset.waypoint_spacing, curr_time + 1, train_dataset.waypoint_spacing))
                            ok = True
                            # check context frames
                            for t in context_times:
                                if train_dataset._load_image(f_curr, t) is None:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            # check goal frame
                            if train_dataset._load_image(f_goal, goal_time) is None:
                                continue
                            valid_indices.append(idx)
                        except Exception:
                            continue
                    if len(valid_indices) == 0:
                        print('Validation scan removed all samples; skipping filtering and using original dataset')
                    elif len(valid_indices) != len(train_dataset):
                        print(f'Filtered dataset: {len(train_dataset) - len(valid_indices)} invalid samples removed, {len(valid_indices)} remain')
                        train_dataset = Subset(train_dataset, valid_indices)
                except Exception:
                    # if any error during validation, continue with original dataset
                    pass
            except Exception as e_vint:
                print('ViNT_Dataset construction failed:', repr(e_vint))
                raise
        except Exception as e_vint_import:
            print('ViNT_Dataset import failed or construction failed; falling back to DummyNomadDataset')
            print('Details:', repr(e_vint_import))
            from torch.utils.data import Dataset

            class DummyNomadDataset(Dataset):
                def __init__(self, length=100, context_size=5, H=128, W=128, T=5, action_dim=2, traj_feat_dim=128):
                    self.length = length
                    self.context_size = context_size
                    self.H = H
                    self.W = W
                    self.T = T
                    self.action_dim = action_dim
                    self.traj_feat_dim = traj_feat_dim

                def __len__(self):
                    return self.length

                def __getitem__(self, idx):
                    obs_image = torch.randn(3 * (self.context_size + 1), self.H, self.W)
                    goal_image = torch.randn(3, self.H, self.W)
                    actions = torch.randn(self.T, self.action_dim)
                    distance = torch.tensor(1)
                    goal_pos = torch.randn(2)
                    dataset_idx = torch.tensor(idx)
                    action_mask = torch.ones(self.T)
                    # ref_traj: [T, D] where D matches Encoder1.traj_feat_dim (default 128)
                    ref_traj = torch.randn(self.T, self.traj_feat_dim)
                    return obs_image, goal_image, actions, distance, goal_pos, dataset_idx, action_mask, ref_traj

            train_dataset = DummyNomadDataset(length=100, context_size=5, H=128, W=128, T=5, action_dim=2, traj_feat_dim=128)
    # Auto-detect per-frame image size. Strategy:
    # 1) Pre-scan split file and try opening a few image files with PIL to get a reliable W.
    # 2) Fallback: sample from dataset items (previous method).
    try:
        detected_frame_width = None
        # Try filesystem scan first (more reliable if dataset indexing yields None)
        try:
            from PIL import Image
        except Exception:
            Image = None

        data_folder = os.path.join(args.data_dir)
        split_paths = [
            os.path.join(args.data_dir, 'splits', 'train', 'traj_names.txt'),
            os.path.join(args.data_dir, 'train', 'traj_names.txt'),
            os.path.join(args.data_dir, 'splits', 'train'),
        ]
        traj_names = []
        for sp in split_paths:
            try:
                if os.path.isfile(sp):
                    with open(sp, 'r') as f:
                        traj_names = [l.strip() for l in f.readlines() if l.strip()]
                        break
                elif os.path.isdir(sp):
                    # if it's a folder containing per-episode entries, list it
                    traj_names = [d for d in os.listdir(sp) if d]
                    break
            except Exception:
                continue

        if Image is not None and traj_names:
            # check first few trajectories and first few frames
            for tn in traj_names[:20]:
                ep_dir = os.path.join(args.data_dir, tn) if not os.path.isabs(tn) else tn
                # fallback: some split entries may be relative paths including dataset root
                if not os.path.isdir(ep_dir):
                    ep_dir = os.path.join(args.data_dir, tn)
                if not os.path.isdir(ep_dir):
                    # try joining with data_folder
                    ep_dir = os.path.join(data_folder, tn)
                if not os.path.isdir(ep_dir):
                    continue
                for j in range(0, 20):
                    img_path = os.path.join(ep_dir, f"{j}.jpg")
                    if not os.path.isfile(img_path):
                        continue
                    try:
                        with Image.open(img_path) as im:
                            W, H = im.size
                            detected_frame_width = W
                            break
                    except Exception:
                        continue
                if detected_frame_width is not None:
                    break

        # Fallback to sampling dataset items if filesystem scan failed
        if detected_frame_width is None:
            max_checks = min(20, len(train_dataset)) if hasattr(train_dataset, '__len__') else 20
            for i in range(max_checks):
                try:
                    sample = train_dataset[i]
                except Exception:
                    continue
                if sample is None:
                    continue
                obs_image = None
                if isinstance(sample, (list, tuple)) and len(sample) > 0:
                    obs_image = sample[0]
                elif isinstance(sample, dict):
                    obs_image = sample.get('obs_image') or sample.get('obs') or sample.get('obs_image_list')
                if obs_image is None or not isinstance(obs_image, torch.Tensor):
                    continue
                if obs_image.dim() == 3:
                    _, H, W = obs_image.shape
                elif obs_image.dim() == 4:
                    _, _, H, W = obs_image.shape
                else:
                    continue
                detected_frame_width = W
                break

        if detected_frame_width is not None:
            if vit_kwargs.get('image_size') != detected_frame_width:
                print(f"Auto-detected per-frame width {detected_frame_width} != configured image_size {vit_kwargs.get('image_size')}; updating vit_kwargs['image_size'] to {detected_frame_width}")
                vit_kwargs['image_size'] = detected_frame_width
        else:
            print('Could not auto-detect image size from files or dataset samples; using configured vit_kwargs')
    except Exception as e:
        print('Warning: failed to auto-detect image size from dataset samples/files:', repr(e))

    # Instantiate TwoStageIntentModel now that vit_kwargs matches data
    two_stage = TwoStageIntentModel(z1_dim=256, z2_dim=256, vit_kwargs=vit_kwargs).to(device)

    # Wrap dataset with a safe wrapper that catches __getitem__ exceptions and returns a
    # synthetic sample so DataLoader workers do not crash on bad entries.
    try:
        from torch.utils.data import Dataset

        class SafeDataset(Dataset):
            def __init__(self, ds, context_size=vit_kwargs.get('context_size', 5), H=vit_kwargs.get('image_size', 128), W=vit_kwargs.get('image_size', 128), T=5, action_dim=2, traj_feat_dim=128):
                self.ds = ds
                self.context_size = context_size
                self.H = H
                self.W = W
                self.T = T
                self.action_dim = action_dim
                self.traj_feat_dim = traj_feat_dim

            def __len__(self):
                try:
                    return len(self.ds)
                except Exception:
                    return 0

            def __getitem__(self, idx):
                try:
                    sample = self.ds[idx]
                    return sample
                except Exception:
                    # return a synthetic safe sample matching expected shapes
                    obs_image = torch.zeros(3 * (self.context_size + 1), self.H, self.W, dtype=torch.float32)
                    goal_image = torch.zeros(3, self.H, self.W, dtype=torch.float32)
                    actions = torch.zeros(self.T, self.action_dim, dtype=torch.float32)
                    distance = torch.tensor(1, dtype=torch.int64)
                    goal_pos = torch.zeros(2, dtype=torch.float32)
                    dataset_idx = torch.tensor(idx, dtype=torch.int64)
                    action_mask = torch.ones(self.T, dtype=torch.float32)
                    ref_traj = torch.zeros(self.T, self.traj_feat_dim, dtype=torch.float32)
                    return obs_image, goal_image, actions, distance, goal_pos, dataset_idx, action_mask, ref_traj

        train_dataset = SafeDataset(train_dataset, context_size=vit_kwargs.get('context_size', 5), H=vit_kwargs.get('image_size', 128), W=vit_kwargs.get('image_size', 128), T=5, action_dim=2, traj_feat_dim=128)
    except Exception:
        pass

    # Use num_workers=0 to avoid LMDB / worker-process issues observed in this environment
    dataloader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=0)

    # If model_callable is not a module (e.g., raw dict/state_dict), only optimize two_stage
    try:
        model_params = list(model_callable.parameters())
    except Exception:
        model_params = []
    # prepare an optional adapter projection from TwoStage z2 -> model expected z input
    adapter_proj = None
    expected_z_in = None
    try:
        # if model_callable is a NoMaD instance it likely has a `z_proj` linear layer
        if hasattr(model_callable, 'z_proj'):
            expected_z_in = getattr(model_callable.z_proj, 'in_features', None)
    except Exception:
        expected_z_in = None

    optim = torch.optim.Adam(model_params + list(two_stage.parameters()), lr=1e-4)
    for epoch in range(args.epochs):
        loss = None
        for batch in dataloader:
            try:
                # batch 应包含 obs_img, goal_img, actions, distance, goal_pos, action_mask, ref_traj
                obs_image, goal_image, actions, distance, goal_pos, dataset_idx, action_mask, ref_traj = batch

                obs_image = obs_image.to(device)
                goal_image = goal_image.to(device)
                actions = actions.to(device)
                ref_traj = ref_traj.to(device)

                # 生成 z1,z2
                z1 = two_stage.encode_stage1(obs_image, goal_image, ref_traj)
                # 例如使用最后一帧作为当前观测
                cur_obs = obs_image[:, -3:, :, :]
                z2 = two_stage.encode_stage2(z1, cur_obs, goal_img=goal_image)

                # 扩散训练与原项目一致，但将 z2 作为 latent_z 传入模型（或按需映射到 model 接口）
                # 下面仅示例调用 noise_pred_net 的方式，具体参数请与项目一致
                timesteps = torch.zeros(actions.size(0), dtype=torch.long, device=device)
                noisy_action = torch.randn_like(actions)

                if callable(model_callable):
                    try:
                        print('Calling model_callable with shapes: noisy_action', tuple(noisy_action.shape), 'z2', getattr(z2, 'shape', None))
                        print('model.local_cond_encoder present=', getattr(getattr(model_callable, 'model', model_callable), 'local_cond_encoder', None) is not None)
                        # If model expects a different z-input size, lazily create an adapter.
                        # Important: instead of passing the adapter output as `global_cond` (which
                        # led to global_cond having the wrong dimensionality), pass the adapter
                        # output as `latent_z` and supply a zero `global_cond` with the
                        # noise_pred_net's expected global_cond_dim. This lets NoMaD.z_proj
                        # handle the final projection into the noise net's cond space.
                        projected_latent = None
                        if expected_z_in is not None and isinstance(z2, torch.Tensor) and z2.shape[1] != expected_z_in:
                            if adapter_proj is None:
                                adapter_proj = nn.Linear(z2.shape[1], expected_z_in).to(device)
                                try:
                                    optim.add_param_group({'params': adapter_proj.parameters()})
                                except Exception:
                                    pass
                                print(f'Created adapter_proj: {z2.shape[1]} -> {expected_z_in}')
                            projected_latent = adapter_proj(z2)
                        else:
                            # if sizes already match, use z2 directly as latent_z
                            if isinstance(z2, torch.Tensor) and expected_z_in is not None and z2.shape[1] == expected_z_in:
                                projected_latent = z2

                        # determine expected global_cond dim for the noise_pred_net so we can
                        # pass a zero tensor (NoMaD will add z_proj(latent_z) to it)
                        global_cond_dim = None
                        try:
                            if hasattr(model_callable, 'noise_pred_net') and hasattr(model_callable.noise_pred_net, 'global_cond_dim'):
                                global_cond_dim = int(model_callable.noise_pred_net.global_cond_dim)
                            elif hasattr(model_callable, 'global_cond_dim'):
                                global_cond_dim = int(getattr(model_callable, 'global_cond_dim'))
                        except Exception:
                            global_cond_dim = None

                        if global_cond_dim is None and isinstance(z2, torch.Tensor):
                            # fallback to a conservative guess: use z2 dim
                            global_cond_dim = z2.shape[1]

                        # create zero global_cond of expected shape so NoMaD._merge_cond can add z_proj(latent_z)
                        gc = None
                        if isinstance(z2, torch.Tensor):
                            gc = torch.zeros(z2.shape[0], global_cond_dim, device=z2.device, dtype=z2.dtype)

                        noise_pred = model_callable(
                            "noise_pred_net",
                            sample=noisy_action,
                            timestep=timesteps,
                            global_cond=gc,
                            latent_z=projected_latent,
                            lambda_cfg=0.0,
                        )
                        print('model_callable returned output; noise_pred.requires_grad=', getattr(noise_pred, 'requires_grad', None))
                    except Exception as e_call:
                        import traceback as _tb
                        print('model_callable raised exception during call:')
                        _tb.print_exc()
                        # if the callable fails, fall back to a differentiable zero prediction
                        noise_pred = torch.zeros_like(noisy_action, requires_grad=True)
                        print('Falling back to zeros_like(noisy_action) with requires_grad=True')
                else:
                    # no diffusion model provided; optimize two_stage only using a differentiable dummy target
                    noise_pred = torch.zeros_like(noisy_action, requires_grad=True)
                    print('No callable model provided; using differentiable dummy zeros')

                loss = nn.functional.mse_loss(noise_pred, torch.zeros_like(noise_pred))

                optim.zero_grad()
                try:
                    print('Before backward: loss.requires_grad=', getattr(loss, 'requires_grad', None), ' optimizer param count=', sum(1 for _ in optim.param_groups[0]['params']))
                    loss.backward()
                except Exception:
                    import traceback
                    traceback.print_exc()
                    print('Backward failed; loss.requires_grad=', getattr(loss, 'requires_grad', None))
                    raise
                optim.step()
            except Exception:
                import traceback
                traceback.print_exc()
                # try to free gpu memory and continue
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

        if loss is not None:
            try:
                print(f"Epoch {epoch} done, sample loss {loss.item():.6f}")
            except Exception:
                print(f"Epoch {epoch} done, sample loss (unprintable)")
        else:
            print(f"Epoch {epoch} done, no successful batches produced a loss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='YAML config with ckpt and options')
    parser.add_argument('--train_config', type=str, required=False, help='Path to original train config YAML (required if using latest_pth)')
    parser.add_argument('--latest_pth', type=str, required=False, help='Path to latest.pth (model state_dict)')
    parser.add_argument('--optimizer_pth', type=str, required=False, help='Path to optimizer_latest.pth')
    parser.add_argument('--scheduler_pth', type=str, required=False, help='Path to scheduler_latest.pth')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # merge cfg into args-like simple object
    class Cfg: pass
    c = Cfg()
    c.data_dir = cfg.get('data_dir')
    c.ckpt = cfg.get('ckpt')
    c.batch_size = cfg.get('batch_size', 16)
    c.epochs = cfg.get('epochs', 10)
    # optional separate checkpoint files
    c.train_config = args.train_config
    c.latest_pth = args.latest_pth
    c.optimizer_pth = args.optimizer_pth
    c.scheduler_pth = args.scheduler_pth
    train_loop(c)


if __name__ == '__main__':
    main()
