import os
import h5py
import torch

from torch.utils.data import DataLoader
import torch.utils.data as data
import pandas as pd
import numpy as np
import openslide


def best_size(natural_mag, size, mag) -> int:
    new_size = natural_mag / mag
    return int(size * new_size)
# faster version
def positional_embeddings_sin_cos(x, y, d_model=2048):
    assert d_model % 2 == 0, "d_model should be even"
    denom = 10000 ** (np.arange(0, d_model // 2, 2) / d_model)
    pe_x = np.zeros(d_model // 2)
    pe_y = np.zeros(d_model // 2)
    #for x
    pe_x[0::2] = np.sin(x / denom)
    pe_x[1::2] = np.cos(x / denom)

    # for y
    pe_y[0::2] = np.sin(y / denom)
    pe_y[1::2] = np.cos(y / denom)

    # Concatenate x and y embeddings and convert to a torch tensor
    positional_embedding = torch.tensor(np.concatenate([pe_x, pe_y]), dtype=torch.float32)

    return positional_embedding


class AttentionDataset(data.Dataset):
    def __init__(self, dataFile='', transform=None, positional_embed = True):
        if isinstance(dataFile,str):
            self.slideData = pd.read_csv(dataFile, sep=",", header=0, index_col=0)
        else:
            self.slideData = dataFile
        if "Unnamed: 0" in self.slideData.columns:
            self.slideData = self.slideData.drop(columns = "Unnamed: 0")
        self.samples = list(self.slideData.index)
        self.labels = list(self.slideData['target'])
        self.files = list(self.slideData['Encoded Path'])
        self.original_slide = list(self.slideData['Original Slide Path'])
        self.positional_embeddings = []

        # # 2d positional embed
        # if positional_embed:
        #     for file_path in self.files:
        #         with h5py.File(file_path, 'r') as hdf5_file:
        #             x_coords = hdf5_file['x'][:]
        #             y_coords = hdf5_file['y'][:]
        #
        #             embeddings = [positional_embeddings_sin_cos(x_coord, y_coord) for x_coord, y_coord in zip(x_coords, y_coords)]
        #             self.positional_embeddings.append(torch.from_numpy(np.array(embeddings)))

    def __getitem__(self, index):
        label = self.labels[index]
        patient_id = None  # Initialize to handle failures
        try:
            slide = openslide.OpenSlide(self.original_slide[index])
            magnification = int(slide.properties.get("openslide.objective-power", 40))  # Default to 40x if missing
        except Exception as e:
            #print(f"Warning: Failed to load slide for index {index}: {e}")
            magnification = 40  

        original_size = best_size(magnification, 256, 20)

        try:
            with h5py.File(self.files[index], 'r') as hdf5_file:
                patient_id = os.path.basename(self.files[index]).replace(".h5", "")
                features = torch.from_numpy(hdf5_file['features'][:])
                x = hdf5_file['x'][:]
                y = hdf5_file['y'][:]
                if "tile_path" in hdfr_file.keys():
                    tile_paths = [path.decode('utf-8') for path in hdf5_file['tile_path'][:]]
                else:
                    tile_paths = [path.decode('utf-8') for path in hdf5_file['tile_paths'][:]]


                scales = 64  
                magnifications = hdf5_file.get('mag', np.array([40]))[0]  

                positional_embed = torch.from_numpy(np.array([
                    positional_embeddings_sin_cos(x_coord, y_coord) for x_coord, y_coord in zip(x, y)
                ]))

            return features, positional_embed, label, x, y, tile_paths, scales, original_size, patient_id

        except Exception as e:
            print(f"Warning: Failed to load HDF5 file for index {index}: {e}")
            return (torch.empty(0), torch.empty(0), label, [], [], [], 64, original_size, "error")
    def __len__(self):
        return (len(self.samples))
class InstanceDataset:
    def __init__(self, bag):
        bag = bag.squeeze(0)
        self.instances = []
        for idx in bag:
            self.instances.append(idx)
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

def instance_dataloader(bag):
    instance_dataset = InstanceDataset(bag)
    return DataLoader(instance_dataset,batch_size =1, shuffle = False)
