import os
import random
import yaml
import argparse

def create_split(data_folder, split_folder, train_ratio=0.9):
    # Get all trajectory directories
    traj_names = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    traj_names.sort()  # Sort to ensure reproducibility
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(traj_names)
    
    num_train = int(len(traj_names) * train_ratio)
    train_trajs = traj_names[:num_train]
    test_trajs = traj_names[num_train:]
    
    print(f"Total trajectories: {len(traj_names)}")
    print(f"Train size: {len(train_trajs)}")
    print(f"Test size: {len(test_trajs)}")
    
    # Write to files
    train_split_path = os.path.join(split_folder, 'train')
    test_split_path = os.path.join(split_folder, 'test')
    
    os.makedirs(train_split_path, exist_ok=True)
    os.makedirs(test_split_path, exist_ok=True)
    
    with open(os.path.join(train_split_path, 'traj_names.txt'), 'w') as f:
        f.write('\n'.join(train_trajs))
        
    with open(os.path.join(test_split_path, 'traj_names.txt'), 'w') as f:
        f.write('\n'.join(test_trajs))
        
    print(f"Splits saved to {split_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--split_folder', type=str, required=True)
    args = parser.parse_args()
    
    create_split(args.data_folder, args.split_folder)
