import os.path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PDBBindPocketWithLigandImageDataset(Dataset):
    """返回的数据格式，适配 EGNN，采用 max_len 的形式
    """

    def __init__(self, dataroot, split_type, split="train", center=True, transform=None, img_transform=None,
                 img_size=224, max_len_pocket=30, ligand_folder="4-view-images-pymol"):
        assert split in ["train", "valid", "val", "test"]
        self.pymol_3d_root = f"{dataroot}/{ligand_folder}"
        self.npz_path = f"{dataroot}/{split_type}/{split}.npz"

        self.transform = transform
        self.img_transform = img_transform
        self.img_size = img_size
        self.max_len_pocket = max_len_pocket

        with np.load(self.npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        self.pdb_name_list = data["names"].tolist()

        # split data based on mask
        self.data = {}
        for (k, v) in data.items():
            if k == 'names' or k == 'receptors' or k == 'affinities':
                self.data[k] = v
                continue

            sections = np.where(np.diff(data['lig_mask']))[0] + 1 if 'lig' in k else \
            np.where(np.diff(data['pocket_mask']))[0] + 1  # 得到 lig 的索引范围
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
                mean = (self.data['lig_coords'][i].sum(0) + self.data['pocket_coords'][i].sum(0)) / (
                            len(self.data['lig_coords'][i]) + len(self.data['pocket_coords'][i]))
                self.data['lig_coords'][i] = self.data['lig_coords'][i] - mean
                self.data['pocket_coords'][i] = self.data['pocket_coords'][i] - mean

    def __len__(self):
        return len(self.data['names'])

    def get_video(self, path_list):
        frame_path_list = path_list
        video = [Image.open(frame_path).convert('RGB') for frame_path in frame_path_list]
        if self.img_transform is not None:
            video = list(map(lambda img: self.img_transform(img).unsqueeze(0), video))
            video = torch.cat(video)
        return video

    def extract_pocket_info(self, data, max_len):
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
            "pocket_coords": pocket_coords_seq,
            "pocket_feats": pocket_one_hot_seq,
            "mask": mask,
            "pocket_frame_id": pocket_frame_id,
            "pocket_pdb_id": pocket_pdb_id,
        }

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.data.items()}
        if self.transform is not None:
            data = self.transform(data)

        # 把口袋数据处理成 max_len 的形式
        data_dict = self.extract_pocket_info(data, max_len=self.max_len_pocket)

        # add image data
        pdb_name = data["names"]

        pymol_3d_video = self.get_video(path_list=[f"{self.pymol_3d_root}/video/{pdb_name}/{filename}" for filename in
                                                   ["x_0.png", "x_180.png", "y_180.png", "z_180.png"]])
        data_dict["3d_video"] = pymol_3d_video  # mask

        # label
        data_dict["affinities"] = data["affinities"]

        return data_dict

    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():

            if prop == 'names' or prop == 'receptors':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_lig_atoms' or prop == 'num_pocket_nodes' or prop == 'num_virtual_atoms':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'mask' in prop:
                out[prop] = torch.stack([x[prop] for x in batch], dim=0)
            elif 'affinities' in prop:
                out[prop] = torch.from_numpy(np.array([x[prop] for x in batch]))
            else:
                out[prop] = torch.stack([x[prop] for x in batch], dim=0)

        out["pocket_feats"] = out["pocket_feats"].long()
        out["mask"] = out["mask"].long().bool()

        return out
