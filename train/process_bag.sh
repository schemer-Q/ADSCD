#!/bin/bash

# 处理bag文件的脚本
python process_bag.py \
    --dataset-name adv_diffusion \
    --input-dir /root/private_data/latent_diffusion_policy/ADSCD/bags/adv_diffusion \
    --output-dir /root/private_data/latent_diffusion_policy/ADSCD/bags/processed_adv_diffusion \
    --sample-rate 4.0