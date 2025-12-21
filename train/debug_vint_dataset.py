# ...existing code...
#!/usr/bin/env python3
"""
调试 ViNT_Dataset 的 __getitem__，定位返回 None 或抛异常的样本/图片。

用法示例:
  # 使用 nomad.yaml
  python train/debug_vint_dataset.py --config ../train/config/nomad.yaml --dataset go_stanford --split train

参数:
  --config : 指向训练用的 yaml（相对 train/ 的路径或绝对路径）
  --dataset: config 中的 datasets key（默认 go_stanford）
  --split  : train 或 test（默认 train）
  --start  : 从哪个索引开始检查（默认 0）
  --max    : 最多检查多少条（默认 全部）
"""
import argparse
import traceback
import yaml
import os
import sys
from pathlib import Path
import importlib

def safe_print(obj, max_len=500):
    try:
        s = repr(obj)
        if len(s) > max_len:
            s = s[:max_len] + "...(truncated)"
        print(s)
    except Exception as e:
        print(f"<repr error: {e}>")

def inspect_dataset_attrs(ds, idx):
    print("==== dataset.__dict__ keys ====")
    keys = list(ds.__dict__.keys())
    print(keys)
    # 查找疑似包含路径/文件/图片的字段
    suspects = [k for k in keys if any(sub in k.lower() for sub in ("img", "path", "file", "frame", "sample", "traj", "meta"))]
    print("==== Suspect fields ====", suspects)
    for k in suspects:
        try:
            v = getattr(ds, k)
            if hasattr(v, "__len__"):
                L = len(v)
            else:
                L = None
            print(f"Field: {k} (len={L})")
            # 尝试打印与 idx 相关的项
            if L is not None and L > 0:
                i = idx % L
                try:
                    item = v[i]
                    safe_print(item, max_len=800)
                except Exception as e:
                    print(f"  <cannot index by {idx}: {e}>")
            else:
                safe_print(v, max_len=400)
        except Exception as e:
            print(f"  <error reading field {k}: {e}>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="config/nomad.yaml", help="path to yaml config")
    ap.add_argument("--dataset", default="go_stanford", help="dataset key in config")
    ap.add_argument("--split", default="train", choices=["train","test"], help="which split to instantiate")
    ap.add_argument("--start", type=int, default=0, help="start index")
    ap.add_argument("--max", type=int, default=None, help="max samples to check")
    ap.add_argument("--single", type=int, default=None, help="check only this single index (overrides start/max)")
    args = ap.parse_args()

    # 加载 config（若有 defaults.yaml 需要合并，可手动先生成完整 config）
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print("Config not found:", cfg_path)
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text())

    if "datasets" not in cfg or args.dataset not in cfg["datasets"]:
        print("Dataset key not in config.datasets")
        print("Available datasets:", list(cfg.get("datasets", {}).keys()))
        sys.exit(1)
    data_cfg = cfg["datasets"][args.dataset]

    # 动态导入 ViNT_Dataset
    try:
        mod = importlib.import_module("vint_train.data.vint_dataset")
        ViNT_Dataset = getattr(mod, "ViNT_Dataset")
    except Exception as e:
        print("Failed to import ViNT_Dataset:", e)
        traceback.print_exc()
        sys.exit(1)

    # 使用与 train.py 中相同的参数来实例化 dataset
    kwargs = dict(
        data_folder=data_cfg.get("data_folder"),
        data_split_folder=data_cfg.get(args.split),
        dataset_name=args.dataset,
        image_size=cfg.get("image_size"),
        waypoint_spacing=data_cfg.get("waypoint_spacing", 1),
        min_dist_cat=cfg["distance"]["min_dist_cat"],
        max_dist_cat=cfg["distance"]["max_dist_cat"],
        min_action_distance=cfg["action"]["min_dist_cat"],
        max_action_distance=cfg["action"]["max_dist_cat"],
        negative_mining=data_cfg.get("negative_mining", True),
        len_traj_pred=cfg.get("len_traj_pred"),
        learn_angle=cfg.get("learn_angle"),
        context_size=cfg.get("context_size"),
        context_type=cfg.get("context_type", "temporal"),
        end_slack=data_cfg.get("end_slack", 0),
        goals_per_obs=data_cfg.get("goals_per_obs", 1),
        normalize=cfg.get("normalize", True),
        goal_type=cfg.get("goal_type", "image"),
    )

    print("Instantiating ViNT_Dataset with:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    ds = ViNT_Dataset(**kwargs)
    print("Dataset len:", len(ds))

    start = args.start
    max_n = args.max if args.max is not None else len(ds)
    if args.single is not None:
        start = args.single
        max_n = args.single + 1

    print(f"Checking indices [{start}, {min(len(ds), max_n)}) one by one (num_workers must be 0 when reproducing DataLoader worker issues).")
    for i in range(start, min(len(ds), max_n)):
        try:
            sample = ds[i]
            if sample is None:
                print(f"Index {i} -> returned None!")
                inspect_dataset_attrs(ds, i)
                break
            # 若 sample 是 tuple/dict，打印简要信息
            if isinstance(sample, dict):
                print(f"Index {i} -> dict keys: {list(sample.keys())}")
            elif isinstance(sample, (list, tuple)):
                print(f"Index {i} -> tuple/list, lengths: {[ (type(x), getattr(x,'shape',None)) for x in sample ]}")
            else:
                print(f"Index {i} -> type {type(sample)}")
        except Exception as e:
            print("=== Exception at index", i, "===")
            traceback.print_exc()
            # 尝试打印 dataset 内可能的路径/图片字段，帮助定位具体文件
            inspect_dataset_attrs(ds, i)
            # 结束调试
            break

if __name__ == "__main__":
    main()
# ...existing code...