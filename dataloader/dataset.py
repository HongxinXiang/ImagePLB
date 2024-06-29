import os.path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PocketWithLigandImageTrajectoryDataset(Dataset):
    """返回的数据格式，适配 EGNN，采用 max_len 的形式
    """
    def __init__(self, dataroot, split="train", center=True, transform=None, img_transform=None, img_size=224,
                 max_len_pocket=30, pocket_folder="pocket", ligand_folder="multi_view_trajectory_video"):
        assert split in ["train", "valid"]
        self.pymol_3d_root = f"{dataroot}/{ligand_folder}"
        self.npz_path = f"{dataroot}/{pocket_folder}/{split}.npz"

        self.pdb_name_list = os.listdir(self.pymol_3d_root)

        self.transform = transform
        self.img_transform = img_transform
        self.img_size = img_size
        self.max_len_pocket = max_len_pocket

        with np.load(self.npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # split data based on mask
        self.data = {}
        for (k, v) in data.items():
            if k == 'names' or k == 'receptors':
                self.data[k] = v
                continue

            sections = np.where(np.diff(data['lig_mask']))[0] + 1 if 'lig' in k else np.where(np.diff(data['pocket_mask']))[0] + 1
            self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            # add number of nodes for convenience
            if k == 'lig_mask':
                self.data['num_lig_atoms'] = torch.tensor([len(x) for x in self.data['lig_mask']])
            elif k == 'pocket_mask':
                self.data['num_pocket_nodes'] = torch.tensor([len(x) for x in self.data['pocket_mask']])

        self.pdb_data_idx_list = defaultdict(list)
        for i, name in enumerate(self.data["names"]):
            pdb_name = name.split("_")[0]
            self.pdb_data_idx_list[pdb_name].append(i)

        if center:
            for i in range(len(self.data['lig_coords'])):
                mean = (self.data['lig_coords'][i].sum(0) + self.data['pocket_coords'][i].sum(0)) / (len(self.data['lig_coords'][i]) + len(self.data['pocket_coords'][i]))
                self.data['lig_coords'][i] = self.data['lig_coords'][i] - mean
                self.data['pocket_coords'][i] = self.data['pocket_coords'][i] - mean

    def __len__(self):
        return len(self.data['names'])

    def get_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img

    def get_video(self, path_list):
        frame_path_list = path_list
        video = [Image.open(frame_path).convert('RGB') for frame_path in frame_path_list]
        if self.img_transform is not None:
            video = list(map(lambda img: self.img_transform(img).unsqueeze(0), video))
            video = torch.cat(video)
        return video

    def load_3d_pymol_image(self, pdb_name, idx_frame):
        root_3d = f"{self.pymol_3d_root}/{pdb_name}"
        axis_info_list = os.listdir(root_3d)
        n_zfill = len(os.listdir(f"{root_3d}/{axis_info_list[0]}")[0].split(".")[0].replace("mov", ""))
        path_list = []
        for axis_info in axis_info_list:
            path = f"{root_3d}/{axis_info}/mov{str(idx_frame).zfill(n_zfill)}.png"
            path_list.append(path)
            if not os.path.exists(path):
                return None
        return self.get_video(path_list)

    def extract_pocket_info(self, data, max_len, prefix=""):
        pocket_coords, pocket_one_hot, pocket_mask, pocket_pdb_mask, num_pocket_nodes = data["pocket_coords"], data[
            "pocket_one_hot"], data["pocket_mask"], data["pocket_pdb_mask"], data["num_pocket_nodes"]
        pocket_frame_id, pocket_pdb_id = torch.unique(pocket_mask), torch.unique(pocket_pdb_mask)

        n_pocket_nodes = num_pocket_nodes.item()
        if max_len >= n_pocket_nodes:  # padding
            pocket_coords_seq = torch.zeros(max_len, 3)
            pocket_one_hot_seq = torch.zeros(max_len)
            mask = torch.zeros(max_len)

            pocket_coords_seq[:n_pocket_nodes] = pocket_coords
            pocket_one_hot_seq[:n_pocket_nodes] = torch.argmax(pocket_one_hot, dim=1).long()
            mask[:n_pocket_nodes] = torch.ones(n_pocket_nodes).long()
        else:  # cut-off
            pocket_coords_seq = pocket_coords[: max_len]
            pocket_one_hot_seq = torch.argmax(pocket_one_hot, dim=1)[: max_len].long()
            mask = torch.ones(max_len).long()

        return {
            f"{prefix}pocket_coords": pocket_coords_seq,
            f"{prefix}pocket_feats": pocket_one_hot_seq,
            f"{prefix}pocket_mask": mask,
            f"{prefix}pocket_frame_id": pocket_frame_id,
            f"{prefix}pocket_pdb_id": pocket_pdb_id,
        }

    def __getitem__(self, idx):
        if idx == len(self) - 1:
            c_data = {key: val[idx - 1] for key, val in self.data.items()}  # current frame
            n_data = {key: val[idx] for key, val in self.data.items()}  # next frame
        else:
            c_data = {key: val[idx] for key, val in self.data.items()}
            n_data = {key: val[idx+1] for key, val in self.data.items()}
            if c_data["names"].split("_frame")[0] != n_data["names"].split("_frame")[0]:
                c_data = {key: val[idx-1] for key, val in self.data.items()}
                n_data = {key: val[idx] for key, val in self.data.items()}

        if self.transform is not None:
            c_data = self.transform(c_data)
            n_data = self.transform(n_data)

        # 检查是否为前后帧的关系
        c_pdb_name, c_idx_frame = c_data["names"].split("_frame")
        n_pdb_name, n_idx_frame = n_data["names"].split("_frame")
        c_idx_frame, n_idx_frame = int(c_idx_frame), int(n_idx_frame)

        try:
            assert c_pdb_name == n_pdb_name and n_idx_frame - c_idx_frame == 1
        except:
            pass

        # 把口袋数据处理成 max_len 的形式
        c_data_dict = self.extract_pocket_info(c_data, max_len=self.max_len_pocket, prefix="c_")
        n_data_dict = self.extract_pocket_info(n_data, max_len=self.max_len_pocket, prefix="n_")

        c_data_dict["c_3d_image"] = self.load_3d_pymol_image(c_pdb_name, c_idx_frame+1)  # 帧是从 1 编号的，所以要加个 1
        n_data_dict["n_3d_image"] = self.load_3d_pymol_image(n_pdb_name, n_idx_frame+1)
        assert c_data_dict["c_3d_image"] is not None and n_data_dict["n_3d_image"] is not None

        data_dict = {**c_data_dict, **n_data_dict}
        return data_dict

    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():
            if prop == 'names' or prop == 'receptors':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_lig_atoms' or prop == 'num_pocket_nodes' or prop == 'num_virtual_atoms':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'pocket_mask' in prop:
                out[prop] = torch.stack([x[prop] for x in batch], dim=0)
            else:
                out[prop] = torch.stack([x[prop] for x in batch], dim=0)

            if "pocket_feats" in prop:
                out[prop] = out[prop].long()

            if "pocket_mask" in prop:
                out[prop] = out[prop].long().bool()
        return out

