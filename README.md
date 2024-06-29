# ImagePLB✨

Official PyTorch-based implementation of Paper "An Image-based Protein-Ligand Binding Representation Learning
Framework via Multi-Level Flexible Dynamics Trajectory Pre-training".



## News!

**[2024/06/28]** Repository installation completed.



## Environments

#### 1. GPU environment

CUDA 11.6

Ubuntu 18.04



#### 2. create conda environment

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



## Pre-Training ImagePLB

#### 1. Pre-training Dataset

| Name                               | Download link                                                | Description                               |
| ---------------------------------- | ------------------------------------------------------------ | ----------------------------------------- |
| multi_view_trajectory_video.tar.gz | [BaiduCloud]()                                               | ligand trajectory with multi-view images. |
| pocket_trajectory.tar.gz           | [OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgUjpCzZnwgp4Inpr?e=ASMVhC) | pocket trajectory with point cloud.       |

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



#### 2. ❄️Direct access to pre-trained ImagePLB

The pre-trained ImagePLB (ImagePLB-P) can be accessed in following table.

| Name           | Download link                                                | Description                                                  |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ImagePLB-P.pth | [OneDrive](https://1drv.ms/u/s!Atau0ecyBQNTgUHGdEk2eovYIGiM?e=q2OPEv) | You can download the ImagePLB-P and put it in the directory: `resumes/`. |



#### 3. 🔥Train your own ImagePLB-P from scratch

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



## 🔥Training ImagePLB on Downstream Tasks

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



- run command in finetune folder for PDBBind:

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



- run command in finetune folder for LEP：

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



## Reproducing Our Results

We provide detailed training logs and corresponding checkpoints, you can easily see more training details from the logs and directly use our trained models for structure-based virtual screening.

| Name             | Download link                                                | Description                                            |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| PDBBind-30       | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTggfNOyNflBKwP42W?e=vrExZf) | The training details of ImagePLB-P on PDBBind-30       |
| PDBBind-60       | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTggjgBxn5vQZV9AfO?e=a8hmfT) | The training details of ImagePLB-P on PDBBind-60       |
| PDBBind-Scaffold | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTggkRSgvkhobN9gYJ?e=PlVC3q) | The training details of ImagePLB-P on PDBBind-Scaffold |
| LEP              | [OneDrive](https://1drv.ms/f/s!Atau0ecyBQNTggBYKwtJl2-6aoUT?e=iQWXe1) | The training details of ImagePLB-P on PDBBind-LEP      |

The files include training logs and checkpoints for training ImagePLB-P with three random seeds (0, 1, 2).



# Reference

If our paper or code is helpful to you, please do not hesitate to point a star for our repository and cite the following content.

```tex
```

