#!/bin/bash

# 训练脚本启动器
# 用途：启动VisualNav-Transformer模型的训练任务

echo "开始启动VisualNav-Transformer训练过程..."

# 设置Python路径（根据实际情况调整）
PYTHON_PATH="python3"

# 训练配置文件路径
CONFIG_FILE="/home/user/diffusion_policy_project/visualnav-transformer/train/config/nomad.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在于 $CONFIG_FILE"
    echo "请检查文件路径是否正确"
    exit 1
fi

# 检查Python是否可用
if ! command -v $PYTHON_PATH &> /dev/null; then
    echo "错误: Python解释器 '$PYTHON_PATH' 未找到"
    echo "请确保Python已正确安装或修改PYTHON_PATH变量"
    exit 1
fi

echo "配置文件: $CONFIG_FILE"
echo "Python路径: $(which $PYTHON_PATH)"
echo "开始执行训练任务..."

# 记录开始时间
START_TIME=$(date)
echo "训练开始时间: $START_TIME"

# 执行训练命令
$PYTHON_PATH train.py -c "$CONFIG_FILE"

# 检查训练命令执行结果
if [ $? -eq 0 ]; then
    echo "训练任务顺利完成!"
else
    echo "错误: 训练任务执行失败，退出代码: $?"
    exit 1
fi

# 记录结束时间
END_TIME=$(date)
echo "训练结束时间: $END_TIME"

echo "VisualNav-Transformer训练任务执行完毕!"