#!/usr/bin/env bash
set -e

# Initialize conda in the current shell (idempotent)
conda init || true

# Reload bash configuration if present
if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  . "$HOME/.bashrc"
fi

# Activate the adscd environment
conda activate /root/private_data/latent_diffusion_policy/env/adscd

echo "Activated conda env: /root/private_data/latent_diffusion_policy/env/adscd"
