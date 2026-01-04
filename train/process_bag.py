import os
import pickle
from PIL import Image
import io
import argparse
import tqdm
import yaml
import rosbag
import numpy as np
import cv2
import rospy
from geometry_msgs.msg import Pose, Twist
import math

# ---------------------- Image processing utils ---------------------- #
def process_compressed_image(msg) -> Image.Image:
    """
    Convert ROS sensor_msgs/CompressedImage to PIL Image
    """
    np_arr = np.frombuffer(msg.data, np.uint8)
    if np_arr.size == 0:
        return None
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_np is None or image_np.size == 0:
        return None
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_np)

def process_raw_image(msg) -> Image.Image:
    """
    Convert ROS sensor_msgs/Image to PIL Image
    """
    try:
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        return Image.fromarray(img)
    except Exception:
        return None

def quaternion_to_yaw(x, y, z, w):
    """
    将四元数转换为偏航角（yaw）
    """
    # 使用公式: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

def process_odom(msg):
    """
    Extract pose data from odometry message
    """
    # 提取位置（只保留x, y，忽略z）
    position = [
        msg.pose.pose.position.x,
        msg.pose.pose.position.y
        # 忽略 z: msg.pose.pose.position.z
    ]
    
    # 提取方向（四元数）
    orientation = [
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    ]
    
    # 计算偏航角（yaw）
    yaw = quaternion_to_yaw(
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    )
    
    # 提取线速度（只保留x, y，忽略z）
    linear_velocity = [
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y
        # 忽略 z: msg.twist.twist.linear.z
    ]
    
    # 提取角速度（只保留z轴，即偏航角速度）
    angular_velocity = [
        msg.twist.twist.angular.z  # 只保留偏航角速度
        # 忽略 x, y: msg.twist.twist.angular.x, msg.twist.twist.angular.y
    ]
    
    return {
        'position': position,  # [x, y]
        'orientation': orientation,  # [x, y, z, w]
        'yaw': yaw,  # 标量
        'linear_velocity': linear_velocity,  # [vx, vy]
        'angular_velocity': angular_velocity,  # [wz]
        'timestamp': msg.header.stamp.to_sec() if hasattr(msg.header.stamp, 'to_sec') else 0.0
    }

# ---------------------- Bag processing ---------------------- #
def get_images_and_odom(bag, img_topics, odom_topics, img_func, odom_func, rate=4.0):
    """
    Read multiple image topics and odom topic from a bag file
    Returns:
        img_data_dict: {topic: [PIL images]}
        odom_data_dict: {topic: [odom dicts]}
    """
    img_data_dict = {t: [] for t in img_topics}
    odom_data_dict = {t: [] for t in odom_topics}

    start_time = bag.get_start_time()
    last_sample_time = start_time
    curr_img_msgs = {t: None for t in img_topics}
    curr_odom_msgs = {t: None for t in odom_topics}

    for topic, msg, t in bag.read_messages(topics=img_topics + odom_topics):
        if topic in img_topics:
            curr_img_msgs[topic] = msg
        elif topic in odom_topics:
            curr_odom_msgs[topic] = msg

        if (t.to_sec() - last_sample_time) >= 1.0 / rate:
            # sample images
            for topic in img_topics:
                if curr_img_msgs[topic] is not None:
                    img = img_func(curr_img_msgs[topic])
                    if img is not None:
                        img_data_dict[topic].append(img)
            # sample odom
            for topic in odom_topics:
                if curr_odom_msgs[topic] is not None:
                    odom = odom_func(curr_odom_msgs[topic])
                    if odom is not None:
                        odom_data_dict[topic].append(odom)
            last_sample_time = t.to_sec()

    # return None if all topics are empty
    if all(len(v) == 0 for v in img_data_dict.values()) or all(len(v) == 0 for v in odom_data_dict.values()):
        return None, None

    return img_data_dict, odom_data_dict

def save_traj_data(odom_list, output_path):
    """
    将 odom 数据保存为 ViNT 期望的格式
    """
    if not odom_list:
        return None
    
    # 提取所有字段
    positions = []
    orientations = []
    yaws = []
    linear_velocities = []
    angular_velocities = []
    timestamps = []
    
    for odom in odom_list:
        positions.append(odom['position'])  # [x, y]
        orientations.append(odom['orientation'])  # [x, y, z, w]
        yaws.append(odom['yaw'])  # 标量
        linear_velocities.append(odom['linear_velocity'])  # [vx, vy]
        angular_velocities.append(odom['angular_velocity'])  # [wz]
        timestamps.append(odom['timestamp'])
    
    # 创建 traj_data 字典，格式与 ViNT 期望的匹配
    traj_data = {
        'position': np.array(positions, dtype=np.float32),  # (N, 2)
        'orientation': np.array(orientations, dtype=np.float32),  # (N, 4)
        'yaw': np.array(yaws, dtype=np.float32),  # (N,)
        'linear_velocity': np.array(linear_velocities, dtype=np.float32),  # (N, 2)
        'angular_velocity': np.array(angular_velocities, dtype=np.float32),  # (N, 1)
        'timestamp': np.array(timestamps, dtype=np.float64),  # (N,)
        'length': len(positions)  # 轨迹长度
    }
    
    # 保存为 pickle
    with open(output_path, 'wb') as f:
        pickle.dump(traj_data, f)
    
    return traj_data

# ---------------------- Main script ---------------------- #
def main(args: argparse.Namespace):
    # load config
    with open("vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(args.output_dir, exist_ok=True)

    # scan bags
    bag_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(args.input_dir)
        for file in files if file.endswith(".bag")
    ]
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]
    print(f"Found {len(bag_files)} bag files in {args.input_dir}")

    # map img_process_func string to actual function
    img_func_map = {
        "process_raw_image": process_raw_image,
        "process_compressed_image": process_compressed_image
    }
    img_func = img_func_map.get(config[args.dataset_name]["img_process_func"], process_raw_image)
    odom_func = process_odom

    traj_counter = 0
    for bag_path in tqdm.tqdm(bag_files, desc="Processing bags"):
        try:
            bag = rosbag.Bag(bag_path)
        except Exception as e:
            print(f"Failed to open {bag_path}: {e}")
            continue

        # get data
        bag_img_data, bag_odom_data = get_images_and_odom(
            bag,
            img_topics=config[args.dataset_name]["imtopics"],
            odom_topics=config[args.dataset_name]["odomtopics"],
            img_func=img_func,
            odom_func=odom_func,
            rate=args.sample_rate
        )
        if bag_img_data is None or bag_odom_data is None:
            print(f"{bag_path} has no valid images/odom, skipping...")
            bag.close()
            continue

        # --------- 创建单独 traj 文件夹 ---------
        traj_name = f"traj_{traj_counter:04d}"
        traj_folder = os.path.join(args.output_dir, traj_name)
        os.makedirs(traj_folder, exist_ok=True)
        traj_counter += 1

        # 保存图像
        image_topic = list(bag_img_data.keys())[0]  # 获取第一个图像topic
        images = bag_img_data[image_topic]
        
        for i, img in enumerate(images):
            if img is not None:
                # 如果需要调整图像大小，可以在这里进行
                # img = img.resize((96, 96))  # 根据配置文件中的 image_size 调整
                img.save(os.path.join(traj_folder, f"{i}.jpg"))

        # 保存轨迹数据 - 使用 ViNT 期望的格式
        odom_topic = list(bag_odom_data.keys())[0]  # 获取第一个odom topic
        odom_list = bag_odom_data[odom_topic]
        
        traj_data_path = os.path.join(traj_folder, "traj_data.pkl")
        traj_data = save_traj_data(odom_list, traj_data_path)
        
        # 同时保存为JSON用于检查
        import json
        json_path = os.path.join(traj_folder, "traj_data.json")
        
        json_data = {}
        for key, value in traj_data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Saved {len(images)} images and {len(odom_list)} odom entries to {traj_folder}")

        bag.close()

# ---------------------- CLI ---------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "-d", type=str, required=True, help="Dataset name in config YAML")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Input directory with rosbag files")
    parser.add_argument("--output-dir", "-o", type=str, default="./dataset", help="Output directory")
    parser.add_argument("--num-trajs", "-n", type=int, default=-1, help="Number of trajectories to process")
    parser.add_argument("--sample-rate", "-s", type=float, default=4.0, help="Sampling rate in Hz")
    args = parser.parse_args()

    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")