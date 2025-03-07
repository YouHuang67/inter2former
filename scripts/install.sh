#!/bin/bash

# 1) Check Python version
python_version=$(python -V 2>&1 | grep -o 'Python [0-9]*.[0-9]*.[0-9]*' | cut -d ' ' -f 2)
if [ "$python_version" != "3.11.0" ]; then
    echo "Warning: Current Python version is not 3.11.0, your version is $python_version."
fi

# 2) Install torch, allowing the user to specify CUDA version or CPU
read -p "Please choose the torch version to install (enter 117 for CUDA 11.7, 118 for CUDA 11.8, or cpu for CPU version): " cuda_version
if [ "$cuda_version" == "117" ]; then
    pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
elif [ "$cuda_version" == "118" ]; then
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
elif [ "$cuda_version" == "cpu" ]; then
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
else
    echo "Error: Invalid input. Please enter 117, 118, or cpu."
    exit 1
fi


# 3) Check for GPU and install xformers
if nvidia-smi &> /dev/null; then
    pip install xformers==0.0.22
else
    echo "Warning: No GPU detected! xformers installation requires GPU and cannot proceed."
    exit 1
fi

# 4) Install timm
pip install timm==0.9.7
pip install einops==0.6.0
pip install numpy==1.24.1
pip install opencv-python==4.6.0.66

# 5) Install mmlab related libraries
pip install -U openmim
mim install mmengine==0.9.0
mim install mmcv==2.0.1

directories=("mmsegmentation" "mmdetection")

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        cd $dir
        pip install -e .
        cd ..
    else
        echo "Warning: Directory $dir does not exist. Skipping installation."
    fi
done

# 6) Install CPP Extensions
cd inter2former/cpp_extension/fast_mask_convert
python setup.py build_ext --inplace
cd ../fast_moe
python setup.py build_ext --inplace
cd ../..
