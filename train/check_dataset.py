# tools/check_dataset.py
import os, pickle, numpy as np
from pathlib import Path

def traj_info(traj_folder):
    p = list(Path(traj_folder).glob("*.jpg"))
    img_count = len(p)
    traj_pkl = Path(traj_folder) / "traj_data.pkl"
    if not traj_pkl.exists():
        return img_count, None
    d = pickle.load(open(traj_pkl, "rb"))
    pos = d["position"]
    yaw = d["yaw"]
    assert pos.shape[0] == yaw.shape[0], f"pos/yaw mismatch {traj_folder}"
    # Convert to a float numpy array safely - some datasets store each position as a small ndarray
    arr = np.asarray(pos)
    if arr.dtype == object:
        try:
            arr = np.stack(arr).astype(float)
        except Exception:
            # fallback: try to coerce element-wise
            arr = np.array([[float(x) for x in p] for p in arr])
    else:
        arr = arr.astype(float)
    mean_spacing = float(np.linalg.norm(arr[1:] - arr[:-1], axis=1).mean()) if len(arr) > 1 else 0.0
    return img_count, mean_spacing

if __name__ == "__main__":
    root="/data/lzq/visualnav-transformer/go_stanford"  # 修改成你的数据目录
    trajs = sorted([os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    spacings=[]
    for t in trajs[:50]:  # 采样检查（你可删掉切片以检查全部）
        img_count, ms = traj_info(t)
        if ms is not None and ms > 0:
            spacings.append(ms)
        print(t, "images:", img_count, "mean spacing:", ms)
    print("Global mean spacing:", np.mean(spacings) if len(spacings)>0 else 0.0)