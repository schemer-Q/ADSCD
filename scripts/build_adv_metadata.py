import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

# Add path to allow imports of vint_train package
# Adjust this path to point to your ADSCD/train directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../train")))

from vint_train.data.vint_dataset import ViNT_Dataset

def extract_adv_info(traj_name, time_step, image=None):
    """
    【关键函数】你需要根据你的数据情况修改这里！
    
    输入:
        traj_name: 轨迹名称 (str)
        time_step: 当前帧 (int)
        image: 图像数据 (Tensor or None) - 如果需要视觉检测则使用
    
    输出:
        is_adv: (bool) 是否为对抗样本
        other_state: (np.array shape [2]) 对手状态，如 [rel_x, rel_y] 或 [bbox_u, bbox_v]
    """
    
    # --- 示例逻辑 1: 基于文件名的规则 ---
    # 如果你的文件名里标记了类别，比如 "adv_scenario_01"
    if "adv" in traj_name or "interaction" in traj_name:
        is_adv = True
        
        # 如果是 Simulator 数据，你可能需要读取对应的 ground truth csv
        # other_state = load_from_log(traj_name, time_step)
        
        # 如果没办法，只能给个随机值先跑通流程 (训练时请务必替换真的)
        # 或者使用视觉检测模型的输出 center point
        other_state = np.array([0.5, 0.5], dtype=np.float32) 
    else:
        is_adv = False
        other_state = np.array([0.0, 0.0], dtype=np.float32)

    return is_adv, other_state

def main():
    parser = argparse.ArgumentParser(description="Generate sidecar metadata for ADSCD training.")
    parser.add_argument('--dataset_name', type=str, default='gazebo', help='Name of dataset in data_config.yaml')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to dataset images')
    parser.add_argument('--split_folder', type=str, required=True, help='Path to data split (train/test)')
    parser.add_argument('--output_path', type=str, default='gazebo_adv_metadata.pkl', help='Output path')
    
    args = parser.parse_args()

    # Initialize ViNT Dataset to iterate easily
    # We only need it to load the index
    print(f"Loading dataset index for {args.dataset_name}...")
    dataset = ViNT_Dataset(
        data_folder=args.data_folder,
        data_split_folder=args.split_folder,
        dataset_name=args.dataset_name,
        image_size=(96, 96),
        waypoint_spacing=1,
        min_dist_cat=0, max_dist_cat=1,
        min_action_distance=1, max_action_distance=1,
        negative_mining=False,
        len_traj_pred=1,
        learn_angle=False,
        context_size=1, # Minimal context for indexing
        goals_per_obs=1
    )
    
    metadata = {}
    print(f"Processing {len(dataset)} samples...")
    
    # Iterate through the dataset
    # We need to access the internal index to get (traj_name, time)
    # dataset.index_to_data contains tuples of info
    
    for i in tqdm(range(len(dataset))):
        # Retrieve internal indexing info
        # ViNT_Dataset implementation detail:
        # self.index_to_data[i] = (traj_name, current_time, max_goal_dist)
        traj_name, curr_time, _ = dataset.index_to_data[i]
        
        # Optional: Load image if you want to run a detector
        # img = dataset._load_image(traj_name, curr_time)
        
        is_adv, other_state = extract_adv_info(traj_name, curr_time)
        
        # Store in dict
        # Key idea: Unique ID for the frame. 
        # Since ViNT dataset constructs samples dynamically (context windows),
        # we index by the 'current' frame.
        key = (traj_name, curr_time)
        metadata[key] = {
            'adv_mask': 1.0 if is_adv else 0.0,
            'other_state': other_state
        }
        
    print(f"Saving metadata to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(metadata, f)
        
    print("Done! You can now use this file in train_adscd.py")

if __name__ == "__main__":
    main()
