#!/bin/bash
# ==========================================
# 脚本名称: conda_env_activator_robust.sh
# 功能描述: 健壮地初始化Conda并激活指定环境
# 更新说明: 解决了conda init后环境激活失败的问题
# ==========================================

set -e  # 遇到错误立即退出脚本

# 添加路径检查函数
check_and_add_env_path() {
    local env_path="/root/private_data/latent_diffusion_policy/env/adscd"
    local parent_dir="/root/private_data/latent_diffusion_policy/env"
    
    # 检查环境目录是否存在
    if [[ ! -d "$env_path" ]]; then
        log_message "错误: 环境路径不存在: $env_path"
        return 1
    fi
    
    # 检查是否在envs_dirs中
    if ! conda config --show envs_dirs | grep -q "$parent_dir"; then
        log_message "添加环境路径到Conda配置: $parent_dir"
        conda config --append envs_dirs "$parent_dir"
    fi
    
    return 0
}

# 配置变量
CONDA_ENV_PATH="/root/private_data/latent_diffusion_policy/env/adscd"
CONDA_SH_PATH="/opt/conda/etc/profile.d/conda.sh"  # 根据您的conda安装路径调整

# 函数：打印带时间戳的日志信息
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 函数：检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 函数：初始化Conda Shell环境
setup_conda_shell() {
    log_message "设置Conda Shell环境..."
    
    # 方法1: 直接source conda的shell脚本 (最可靠的方法)
    if [[ -f "$CONDA_SH_PATH" ]]; then
        log_message "直接加载Conda Shell脚本: $CONDA_SH_PATH"
        source "$CONDA_SH_PATH"
        return 0
    else
        log_message "警告: 未找到conda.sh脚本，尝试其他方法" >&2
    fi
    
    # 方法2: 尝试初始化bash
    if command_exists conda; then
        log_message "运行conda init bash..."
        conda init bash >/dev/null 2>&1 || true
        
        # 重新加载bashrc
        if [[ -f ~/.bashrc ]]; then
            log_message "重新加载.bashrc..."
            source ~/.bashrc
        fi
    fi
    
    # 方法3: 最后尝试，使用conda shell.bash hook
    if command_exists conda; then
        log_message "使用conda shell.bash hook..."
        eval "$(conda shell.bash hook)"
    fi
}

# 主函数
main() {
    log_message "开始初始化Conda环境..."
    
    # 1. 检查conda是否可用
    if ! command_exists conda; then
        log_message "错误: 未找到conda命令，请确保Conda已正确安装" >&2
        exit 1
    fi
    
    # 2. 设置Conda Shell环境
    setup_conda_shell
    
    # 3. 检查是否能够激活base环境来验证初始化成功
    if ! conda activate base >/dev/null 2>&1; then
        log_message "错误: Conda Shell初始化失败，无法激活环境" >&2
        exit 1
    fi
    log_message "Conda Shell初始化成功"
    
    # 4. 停用base环境，准备激活目标环境
    conda deactivate 2>/dev/null || true

    # 4.5 在激活环境前添加检查
    if check_and_add_env_path; then
        log_message "尝试激活Conda环境: $CONDA_ENV_PATH"
        conda activate "$CONDA_ENV_PATH"
        # ... 其余代码 ...
    else
        log_message "错误: 环境路径检查失败"
        exit 1
    fi
    
    # 5. 激活指定Conda环境
    log_message "尝试激活Conda环境: $CONDA_ENV_PATH"
    
    # 检查环境路径是否存在
    if conda env list | grep -q "$CONDA_ENV_PATH"; then
        if conda activate "$CONDA_ENV_PATH"; then
            log_message "成功激活Conda环境: $CONDA_ENV_PATH"
            
            # 验证环境是否激活成功
            if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
                log_message "当前激活的环境: $CONDA_DEFAULT_ENV"
                log_message "Python路径: $(which python 2>/dev/null || echo '未找到')"
            else
                log_message "警告: 环境可能未正确激活" >&2
            fi
        else
            log_message "错误: 环境激活失败" >&2
            exit 1
        fi
    else
        log_message "错误: 指定的Conda环境不存在: $CONDA_ENV_PATH" >&2
        log_message "可用的Conda环境列表:" 
        conda env list
        exit 1
    fi
    
    log_message "Conda环境初始化完成"
}

# 执行主函数
main "$@"