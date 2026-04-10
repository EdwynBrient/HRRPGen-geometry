import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
try:
    from .utils import *
except ImportError:  # pragma: no cover
    from utils import *
import torch.nn.functional as F
import torch
from torchvision import transforms
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import skimage
import timm
from tqdm import tqdm
import pandas as pd

selectRP = [str(i) for i in range(200)]

class RP_ImageDataset(Dataset):
    def __init__(self, config, path_rp="../data/all_df", nb_img=2):
        self.nb_img = nb_img
        self.df = pd.read_csv(path_rp)
        count_mmsi = self.df.mmsi.value_counts()
        self.df = self.df.sort_values(["length", "width"], axis=0)
        self.df = self.df[~(self.df.length==0)]
        self.df = self.df[self.df.mmsi.map(count_mmsi) >= 30]
        self.df = translate_zero_gaussian(self.df)
        self.config=config
        self.base = "../data/"
        self.img_path = "predictions/"
        self.lim_data = config["lim_data"]
        self.orientation_pad_dict = {0: 80, 1: 60, 2: 80, 3: 100}
        self.orientation_rad_dict = {0: [[3*np.pi/4-np.pi/8, 3*np.pi/4+np.pi/8],[7*np.pi/4-np.pi/8, 7*np.pi/4+np.pi/8]],
                                     1: [[7*np.pi/4+np.pi/8, np.pi/4-np.pi/8], [np.pi-np.pi/8, np.pi+np.pi/8]],
                                     2: [[np.pi/4-np.pi/8, np.pi/4+np.pi/8], [5*np.pi/4-np.pi/8, 5*np.pi/4+np.pi/8]],
                                     3: [[np.pi/2-np.pi/8, np.pi/2+np.pi/8], [6*np.pi/4-np.pi/8, 6*np.pi/4+np.pi/8]]}
        
        
        self.processor  = transforms.Compose([
        transforms.Resize(224),  # Resize shorter side
        transforms.CenterCrop(224),  # Center crop
        transforms.ToTensor(),  # Convert to tensor
        ])
        
        self.df = self.df.iloc[:int(self.lim_data)] if int(self.lim_data)!=0 else self.df
        if self.config["conditionned"]["type"] != "scalars":
            self.all_masks = self.select_RP_with_image()
        self.old_va = self.df.viewing_angle.copy()
        self.min_rp, self.max_rp = self.normalize_df()
        self.preprocess_vars()
        
    def select_RP_with_image(self):
        df_sel_img_list = []
        all_masks = {}
        for mmsi_folder in os.listdir(os.path.join(self.base, self.img_path)):
            mmsi = int(mmsi_folder)
            mmsi_path = os.path.join(os.path.join(self.base, self.img_path), mmsi_folder)
            if os.path.isdir(mmsi_path):
                list_img = [int(i[0]) for i in os.listdir(mmsi_path)]
                orientation = [i-4 for i in range(4, 8) if i in list_img]
                for orient in orientation:
                    orient_rp = (orient-2)%4
                    df_temp = self.df[self.df.mmsi == mmsi]
                    df_va = df_temp["viewing_angle"]
                    rng1, rng2 = self.orientation_rad_dict[orient_rp][0], self.orientation_rad_dict[orient_rp][1]
                    mask = mask_dfva_in_rng(df_va, rng1, rng2, orient_rp)
                    df_sel_img_list.append(df_temp[mask])
                    if len(df_temp[mask]) > 0:
                        rngpng = [orient, orient+4] if orient != 0 else [orient+4, orient+8] 
                        for png in rngpng:
                            mask = Image.open(self.base + "predictions/" + mmsi_folder + "/" + str(png) + ".png")
                            mask = pad_to_square(mask, fill=0.)
                            mask = self.processor(mask)
                            if mmsi not in all_masks:
                                all_masks[mmsi] = {}
                            all_masks[mmsi][str(png)] = mask    
                    
        self.df = pd.concat(df_sel_img_list)
        return all_masks
    
    def normalize_df(self):
        self.df.length = self.df.length / self.df.length.max()
        self.df.width = self.df.width / self.df.width.max()
        self.df.viewing_angle = self.df.viewing_angle / self.df.viewing_angle.max()
        min_rp, max_rp = 0., self.df[selectRP].max().max()
        self.df[selectRP] = (self.df[selectRP]- min_rp) / (max_rp - min_rp)
        self.df[selectRP] = self.df[selectRP] * 2 - 1
        return min_rp, max_rp

    def get_key_for_angle(self, angle):
        for key, intervals in self.orientation_rad_dict.items():
            for interval in intervals:
                start, end = interval
                # Handle the circular nature of angles (wrap around at 2π)
                if start < end:
                    if start <= angle <= end:
                        return key
                else:  # Interval wraps around 2π (e.g., 7π/4 to π/4)
                    if start <= angle or angle <= end:
                        return key
        return None

    def __len__(self):
        return len(self.df)
    
    def preprocess_vars(self):
        self.viewing_angles = torch.tensor(self.df.viewing_angle.values * self.old_va.max(), dtype=torch.float32)
        self.lengths = torch.tensor(self.df.length.values, dtype=torch.float32)
        self.widths = torch.tensor(self.df.width.values, dtype=torch.float32)
        self.hrrps = torch.tensor(np.array(self.df[selectRP]), dtype=torch.float32)
        self.mmsis = self.df.mmsi.unique()
    
    def select_masks(self, mmsi, va_list):
        """
        Retrieve masks for a given MMSI and one or multiple angle values (va).
        
        :param mmsi: The MMSI identifier of the ship.
        :param va_list: A single angle value or a list of angles.
        :return: A tuple of masks if a single angle is provided, or a list of tuples for multiple angles.
        """
        if not isinstance(va_list, (list, tuple, np.ndarray)):
            va_list = [va_list]  # Convert single va to a list
        
        mask_pairs = []
        for va in va_list:
            orientation = self.get_key_for_angle(va)
            orientation_perp = (orientation - 2) % 4
            png_1, png_2 = orientation_perp, orientation_perp + 4
            if orientation_perp == 0:
                png_1, png_2 = png_1 + 4, png_2 + 4
            
            mask1 = self.all_masks[mmsi].get(str(png_1))
            mask2 = self.all_masks[mmsi].get(str(png_2))
            
            mask_pairs.append((mask1, mask2))
        
        return mask_pairs if len(mask_pairs) > 1 else mask_pairs[0]

    def __getitem__(self, idx):
        hrrp = self.hrrps[idx]
        va = self.viewing_angles[idx:idx+1]
        length = self.lengths[idx:idx+1]
        width = self.widths[idx:idx+1]
        mmsi = self.df.mmsi.iloc[idx]

        vars = torch.cat([hrrp.unsqueeze(0), va.unsqueeze(0), length.unsqueeze(0), width.unsqueeze(0)], dim=1)
        if self.config["conditionned"]["type"] == "scalars":
            return vars, torch.Tensor([idx]).type(torch.uint32)
        # Image processing
        mask1, mask2 = self.select_masks(mmsi, va)
        masks = torch.stack([mask1, mask2])        
        return masks, vars, torch.Tensor([idx]).type(torch.uint32)


if __name__ == "__main__":

    # # Exemple d'utilisation
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((224, 224))])
    CONFIG_PATH = os.path.join("./config", "configs_v100/cf0_drop1.yaml")
    config = load_config(CONFIG_PATH)
    config["lim_data"] = 10000

    dataset = RP_ImageDataset(config=config)
    mmsi1 = dataset.mmsis[0]
    png = list(dataset.embeddings_stock[mmsi1].keys())[0]
    emb1, emb2 = dataset.embeddings_stock[mmsi1][png], dataset.embeddings_stock[mmsi1][str(int(png)+4)]
    print(emb1.shape, emb2.shape)
    print(dataset[0][0].shape, dataset[0][1].shape)


    # RP = vars[:, 0, :200].float()
    # print(vars.shape)
    # vars = torch.concat([vars[:, :, -1], vars[:, :, -2], vars[:, :, -3]], dim=1).float()
    # print(vars.shape)

    # images_np0 = img1.detach().cpu().numpy()
    # images_np0 = ((images_np0 + 1) / 2 * 255.).astype(int)
    # images_np0 = np.transpose(images_np0, (0, 2, 3, 1))  # Change shape to (batch_size, height, width, channels)
    # images_np1 = img2.detach().cpu().numpy()
    # images_np1 = ((images_np1 + 1) / 2 * 255.).astype(int)
    # images_np1 = np.transpose(images_np1, (0, 2, 3, 1))  # Change shape to (batch_size, height, width, channels)

    # for i in range(10):
    #     plt.imshow(images_np0[i])
    #     plt.show()

    #     plt.imshow(images_np1[i])
    #     plt.show()
    #     labels_np = [label.detach().cpu().numpy()*6.28 for label in vars]
    #     print(labels_np[i][2], dataset.get_key_for_angle(labels_np[i][2]))

    # i=0
    # for label in labels_np:
    #     if i!=0:
    #         print(label[0])
    #     i+=1
    #
    # print(get_means_stddevs(dataloader))
    # x = []
    # for i in range(batch.shape[1]):
    #     img1, img2 = batch[0, i, :], batch[1, i, :]
    #     x1, x2 = img1.squeeze(), img2.squeeze()
    #     print(x1.shape, x2.shape)
    #     x1, x2 = np.array(x1), np.array(x2)
    #     x1, x2 = np.transpose(x1, (1, 2, 0)), np.transpose(x2, (1, 2, 0))
    #     x.append([x1, x2])
    #
    # for x1, x2 in x:
    #     plt.imshow(x1)
    #     plt.show()
    #     plt.imshow(x2)
    #     plt.show()
