#!/bin/bash

# Setup Virtual Environment
echo "Setting up Virtual Environment"
module load cray-python
python -m venv ./my-venv --system-site-packages
cd my-venv
. bin/activate

# PyTorch
echo "Installing PyTorch"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

module load amd-mixed/5.6.0
module load PrgEnv-cray

# mpi4py
echo "Installing mpi4py"
module load craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CRAY_MPICH_ROOTDIR}/gtl/lib"
echo ${LD_LIBRARY_PATH}

MPICC=CC python -m pip install --ignore-installed --no-cache-dir mpi4py

# Flash Attention
echo "Installing Flash Attention"
git clone https://github.com/ROCmSoftwarePlatform/flash-attention
cd flash-attention
vi setup.py -c ':%s/c++20/c++17/g' -c ':wq'
CC=cc CXX=CC PYTORCH_ROCM_ARCH='gfx90a' GPU_ARCHS='gfx90a' pip install -v .

# Apex
echo "Installing Apex"
cd ..
git clone https://github.com/ROCmSoftwarePlatform/apex
cd apex
git checkout release/1.1.0
CC=cc CXX=CC PYTORCH_ROCM_ARCH='gfx90a' GPU_ARCHS='gfx90a' python setup.py install --cpp_ext --cuda_ext

# RCCL Plugin
echo "Installing RCCL Plugin"
cd ..
git clone https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
cd aws-ofi-rccl
module load libtool
./autogen.sh
CC=cc CXX=CC ./configure --with-libfabric=/opt/cray/libfabric/1.15.0.0 --with-hip=/opt/rocm-5.6.0/ --with-rccl="$(dirname "$(pwd)")"/lib/python3.9/site-packages/torch/lib/ --prefix="$(dirname "$(pwd)")"/aws-ofi-rccl/build/
CC=cc CXX=CC make -j install

cd ..

# AxoNN
echo "Installing AxoNN"
git clone https://github.com/axonn-ai/axonn.git
cd axonn
pip install -e .

cd ..

# Megatron-AxoNN
echo "Installing Megatron-AxoNN"
git clone https://github.com/axonn-ai/Megatron-AxoNN.git

pip install regex

echo "Done!"

