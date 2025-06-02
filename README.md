# ImagePLB 🐘

Official PyTorch-based implementation of Paper:

**"An Image-based Protein-Ligand Binding Representation Learning Framework via Multi-Level Flexible Dynamics Trajectory Pre-training"**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
<a href="https://github.com/HongxinXiang/ImagePLB/blob/master/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/HongxinXiang/ImagePLB?style=flat-square"></a>
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/HongxinXiang/ImagePLB?style=flat-square">
<a href="" target='_blank'><img src="https://visitor-badge.laobi.icu/badge?page_id=HongxinXiang.ImagePLB-X&left_color=gray&right_color=orange"></a>



## 📌 Table of Contents

- [🚀 News](#-news)
- [🛠️ Installation & Environment Setup](#️-installation--environment-setup)
- [🖼️ Data Preprocessing](#️-data-preprocessing)
- [🧪 Pre-training ImagePLB](#-pre-training-imageplb)
- [🎯 Fine-tuning on Downstream Tasks](#-fine-tuning-on-downstream-tasks)
- [📊 Reproducing Our Results](#-reproducing-our-results)
- [📚 Citation](#-citation)



---



## 🚀 News

- **[2024/06/28]** Repository setup completed. Code and instructions released.



## 🛠️ Installation & Environment Setup

### 1. Hardware/Software Environment

- GPU with CUDA 11.6
- Ubuntu 18.04


### 2. Setup with Conda

```bash
# create conda env
conda create -n ImagePLB python=3.9
conda activate ImagePLB
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# install environment
pip install rdkit
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install biopython==1.79
pip install easydict
pip install tqdm
pip install timm==0.6.12
pip install tensorboard
pip install scikit-learn
pip install setuptools==59.5.0
pip install pandas
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
pip install torch-geometric==1.6.0
pip install dgl-cu116
pip install ogb
pip install seaborn
conda install openbabel -c conda-forge
pip install einops
```



## 🖼️ Data Preprocessing

We use PyMOL to genearte multi-view ligand images from molecular conformations. Here is the PyMOL script to get the multi-view ligand images, you can run it in the PyMOL command:

<details>
<summary>Click here for the code!</summary>

```bash
sdf_filepath=demo.sdf  # sdf file path of ligand
rotate_direction=x
rotate=0
save_img_path=demo_frame.png
load $sdf_filepath;bg_color white;set stick_ball,on;set stick_ball_ratio,3.5;set stick_radius,0.15;set sphere_scale,0.2;set valence,1;set valence_mode,0;set valence_size, 0.1;rotate $rotate_direction, $rotate;save $save_img_path;quit;
```

</details>

Note that we used 4 views by setting the following parameters:

- rotate_direction=x; rotate=0
- rotate_direction=x; rotate=180
- rotate_direction=y; rotate=180
- rotate_direction=z; rotate=180

Of course, to save you time on data preprocessing, we also provide download links for all data for your free access.



## 🧪 Pre-training ImagePLB

#### 1. Pre-training Dataset

| Name                                  | Download link                                                | Description                                                  |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| multi_view_trajectory_video.tar.gz    |                                                              |                                                              |
| multi_view_trajectory_video_1%.tar.gz | [BaiduCloud](https://pan.baidu.com/s/1ijVSX4ORfYQvAgxpcGzjEg?pwd=khbx) | ligand trajectory with 1% multi-view images. (only #1 frame is multi-view images) |
| pocket.tar.gz                         | [OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgUjpCzZnwgp4Inpr?e=ASMVhC) | pocket trajectory with 3D graphs.                            |

Please download all data listed above and put it in `datasets/pre-training/MISATO/processed/` if you want to pre-train ImagePLB from scratch.

The directory is organized in the following format:

```
datasets/pre-training/MISATO/processed/
+---pocket
|   |   train.npz
|   |
+---multi_view_trajectory_video
|   +---1A0Q
|   |   +---x_0
|   |   |   mov0001.png
|   |   |   mov0002.png
|   |   |   ...
|   |   +---x_180
|   |   |   mov0001.png
|   |   |   mov0002.png
|   |   |   ...
|   |   +---y_180
|   |   |   mov0001.png
|   |   |   mov0002.png
|   |   |   ...
|   |   +---z_180
|   |   |   mov0001.png
|   |   |   mov0002.png
|   |   |   ...
```



#### 2. Download Pretrained Model

The pre-trained ImagePLB (ImagePLB-P) can be accessed in following table.

| Name           | Download link                                                | Description                                                  |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ImagePLB-P.pth | [OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgUHGdEk2eovYIGiM?e=q2OPEv) | You can download the ImagePLB-P and put it in the directory: `resumes/`. |



#### 3. Pre-train ImagePLB-P from Scratch

If you want to pre-train your own ImagePLB-P, see the command below.

Usage:

```bash
usage: pretrain_ImagePLB.py [-h] [--dataroot DATAROOT] [--workers WORKERS]
                            [--model_name MODEL_NAME]
                            [--max_len_pocket MAX_LEN_POCKET] [--center]
                            [--n_dim_graph N_DIM_GRAPH] [--lr LR]
                            [--momentum MOMENTUM]
                            [--weight-decay WEIGHT_DECAY] [--weighted_loss]
                            [--runseed RUNSEED] [--start_epoch START_EPOCH]
                            [--epochs EPOCHS] [--batch BATCH]
                            [--imageSize IMAGESIZE] [--resume RESUME]
                            [--n_ckpt_save N_CKPT_SAVE]
                            [--n_batch_step_optim N_BATCH_STEP_OPTIM]
                            [--lambda_next_mol LAMBDA_NEXT_MOL]
                            [--lambda_next_pocket LAMBDA_NEXT_POCKET]
                            [--lambda_next_complex LAMBDA_NEXT_COMPLEX]
                            [--log_dir LOG_DIR] [--tb_step_num TB_STEP_NUM]
```



run command in pretrain folder to pre-train ImagePLB:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain_ImagePLB.py \
	--workers 16 \
	--batch 128 \
	--epochs 30 \
	--lr 0.001 \
	--dataroot ../datasets/pre-training/MISATO/processed \
	--log_dir ./experiments/pretrain_ImagePLB \
	--weighted_loss
```



## 🎯 Fine-tuning ImagePLB on Downstream Tasks

### 1. Datasets

All downstream task data is publicly accessible below:

| Datasets | Links                                                        | Description                                         |
| -------- | ------------------------------------------------------------ | --------------------------------------------------- |
| PDBBind  | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTgUv2mJseSRgKZcU7?e=5a2CpG) | Including PDBBind-30, PDBBind-60, PDBBind-Scaffold. |
| LEP      | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTgVBQrrpS7g-J9bzD?e=XJvVTK) | Dataset of ligand efficacy prediction.              |



⚠️Please download the dataset provided above and organize the directory as follows:

```
datasets/fine-tuning/
+---pdbbind
|   +---ligand
|   |   +---1a4k
|   |   |   x_0.png
|   |   |   x_180.png
|   |   |   y_180.png
|   |   |   z_180.png
|   +---30
|   |   |   train.npz
|   |   |   valid.npz
|   |   |   test.npz
|   +---60
|   |   |   train.npz
|   |   |   valid.npz
|   |   |   test.npz
|   +---scaffold
|   |   |   train.npz
|   |   |   valid.npz
|   |   |   test.npz
+---lep
|   +---ligand
|   |   +---Lig2__6BQG__6BQH
|   |   |   x_0.png
|   |   |   x_180.png
|   |   |   y_180.png
|   |   |   z_180.png
|   +---protein
|   |   |   train.npz
|   |   |   val.npz
|   |   |   test.npz
```



### 2. Run Fine-tuning

- run command in finetune folder **for PDBBind**:

```bash
python pdbbind.py \
	--batch 32 \
	--epochs 20 \
	--lr 0.0001 \
	--egnn_dropout 0.3 \
	--predictor_dropout 0.3 \
	--dataroot ../datasets/fine-tuning/pdbbind \
	--split_type scaffold \
	--resume ../resumes/ImagePLB-P.pth \
	--log_dir ./experiments/pdbbind/scaffold/rs0/ \
	--runseed 0 \
	--dist-url tcp://127.0.0.1:12312
```



- run command in finetune folder **for LEP**：

```bash
python lep.py \
	--batch 32 \
	--epochs 100 \
	--lr 0.0001 \
	--dataroot ../datasets/fine-tuning/lep \
	--split_type protein \
	--egnn_dropout 0.5 \
	--predictor_dropout 0.5 \
	--resume ../resumes/ImagePLB-P.pth \
	--log_dir ./experiments/lep/rs0/ \
	--runseed 0 \
	--dist-url tcp://127.0.0.1:12345
```



## 📊 Reproducing Our Results

We provide detailed training logs and corresponding checkpoints, you can easily see more training details from the logs and directly use our trained models for structure-based virtual screening.

| Name             | Download link                                                | Description                                           |
| ---------------- | ------------------------------------------------------------ | ----------------------------------------------------- |
| PDBBind-30       | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTggfNOyNflBKwP42W?e=vrExZf) | The training details of ImagePLB-P on PDBBind-30      |
| PDBBind-60       | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTggjgBxn5vQZV9AfO?e=a8hmfT) | The training details of ImagePLB-P on PDBBind-60      |
| PDBBind-Scaffold | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTggkRSgvkhobN9gYJ?e=PlVC3q) | The training details of ImagePLB-P on PDBBind-Scaffold |
| LEP              | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTggBYKwtJl2-6aoUT?e=iQWXe1) | The training details of ImagePLB-P on LEP      |

The files include training logs and checkpoints for training ImagePLB-P with three random seeds (0, 1, 2).



# 📚 Citation

If you find this repository helpful, please consider citing our work and starring 🌟 the repository.

```tex

```



