#!/usr/bin/env bash
set -euo pipefail

# Install into a dedicated environment, without changing platform or global config.
dtiuie_mode="${1:-legacy}"
dtiuie_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dtiuie_pip_index="${DTIUIE_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
dtiuie_conda_channel="${DTIUIE_CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"

case "$dtiuie_mode" in
    legacy)
        dtiuie_python_version="3.9.16"
        dtiuie_torch_index="https://download.pytorch.org/whl/cu116"
        dtiuie_torch_packages=("torch==1.13.1+cu116" "torchvision==0.14.1+cu116")
        ;;
    modern)
        dtiuie_python_version="3.9.25"
        dtiuie_torch_index="https://download.pytorch.org/whl/cu128"
        dtiuie_torch_packages=("torch==2.7.1+cu128" "torchvision==0.22.1+cu128")
        ;;
    *)
        printf 'Usage: bash setup_env.sh legacy|modern [environment-prefix]\n' >&2
        exit 2
        ;;
esac

dtiuie_env_prefix="${2:-/root/autodl-tmp/envs/dtiuie-$dtiuie_mode}"
if ! command -v conda >/dev/null 2>&1; then
    printf 'conda is missing; choose an AutoDL Miniconda/PyTorch image.\n' >&2
    exit 2
fi
if [[ ! -d "$dtiuie_env_prefix/conda-meta" ]]; then
    if [[ -e "$dtiuie_env_prefix" ]]; then
        printf 'Refusing to replace a non-Conda directory: %s\n' "$dtiuie_env_prefix" >&2
        exit 2
    fi
    conda create --prefix "$dtiuie_env_prefix" --override-channels \
        --channel "$dtiuie_conda_channel" "python=$dtiuie_python_version" pip --yes
fi

dtiuie_env_python="$dtiuie_env_prefix/bin/python"
"$dtiuie_env_python" -m pip install --index-url "$dtiuie_pip_index" \
    --extra-index-url "$dtiuie_torch_index" \
    "${dtiuie_torch_packages[@]}" "numpy==1.23.5" "Pillow==9.4.0"
"$dtiuie_env_python" -m pip install --index-url "$dtiuie_pip_index" \
    --requirement "$dtiuie_script_dir/requirements-inference.txt"
"$dtiuie_env_python" -m pip check
"$dtiuie_env_python" -m pip freeze > "$dtiuie_env_prefix/dtiuie-packages.txt"
printf 'Prepared environment: %s\n' "$dtiuie_env_prefix"
printf 'Run GPU verification in GPU mode: %s %s/verify_gpu.py\n' "$dtiuie_env_python" "$dtiuie_script_dir"
