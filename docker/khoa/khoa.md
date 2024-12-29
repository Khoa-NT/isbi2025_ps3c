# Khoa's environment

## nvidia-smi
```bash
Mon Dec 30 02:21:11 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A6000               Off |   00000000:4F:00.0 Off |                    0 |
| 30%   36C    P2             71W /  300W |       1MiB /  46068MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```

## nvcc --version
```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

## python --version
```bash
Python 3.11.9
```

## pip freeze > requirements.txt
```bash
pip freeze > requirements.txt
```

## conda env export > environment.yml
```bash
conda env export > environment.yml
```

## lsb_release -a
```bash
Distributor ID:	Ubuntu
Description:	Ubuntu 22.04.1 LTS
Release:	22.04
Codename:	jammy
```

## cat /etc/os-release
```bash
PRETTY_NAME="Ubuntu 22.04.1 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.1 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```

## nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv
```bash
name, driver_version, memory.total [MiB]
NVIDIA RTX A6000, 550.127.05, 46068 MiB
```

## cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
CuDNN is installed concurrently with PyTorch.
```bash
cuda-cudart               12.4.127                      0    nvidia
cuda-cupti                12.4.127                      0    nvidia
cuda-libraries            12.4.1                        0    nvidia
cuda-nvrtc                12.4.127                      0    nvidia
cuda-nvtx                 12.4.127                      0    nvidia
cuda-opencl               12.6.77                       0    nvidia
cuda-runtime              12.4.1                        0    nvidia
cuda-version              12.6                          3    nvidia
pycuda                    2024.1.2                 pypi_0    pypi
pytorch                   2.5.1           py3.11_cuda12.4_cudnn9.1.0_0    pytorch
pytorch-cuda              12.4                 hc786d27_7    pytorch
pytorch-mutex             1.0                        cuda    pytorch
```

## dpkg -l | grep cudnn
CuDNN is installed concurrently with PyTorch.

