import os
import h5py
import torch

from torch.utils.data import DataLoader
import torch.utils.data as data
import pandas as pd
import numpy as np
import openslide
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy.spatial.distance import cdist
import jax.numpy as jnp

# spatial neighborhoods

def aggregate_local_features_euclidean(embeddings, coords, knn_indices):
    coords = jnp.array(coords)
    embeddings = jnp.array(embeddings)
    
    coord_i_all = coords[:, None, :]                        # (N, 1, 2)
    neighbor_coords = coords[knn_indices]                   # (N, k, 2)
    distances = jnp.linalg.norm(coord_i_all - neighbor_coords, axis=-1)  # (N, k)

    sigma = jnp.sqrt(1 / (1 * 1e-6))
    weights = (1 / jnp.sqrt(2 * jnp.pi * sigma**2)) * jnp.exp(-1e-6 * distances**2)
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)          # (N, k)

    neighbors = embeddings[knn_indices]                    # (N, k, D)
    weighted = weights[..., None] * neighbors              # (N, k, D)
    aggregated = jnp.sum(weighted, axis=1)                 # (N, D)
    
    return aggregated

def compute_knn(coords, k=5):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    _, indices = nbrs.kneighbors(coords)
    return indices[:, 1:]

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
    def __init__(self, dataFile='', transform=None, positional_embed = True, type_embed = None, mode = "train"):
        if isinstance(dataFile,str):
            self.slideData = pd.read_csv(dataFile, sep=",", header=0, index_col=0)
        else:
            self.slideData = dataFile
        if "Unnamed: 0" in self.slideData.columns:
            self.slideData = self.slideData.drop(columns = "Unnamed: 0")
        self.samples = list(self.slideData.index)
        self.labels = list(self.slideData['target'])
        self.files = list(self.slideData['Encoded Path'])
        if 'Original Slide Path' in self.slideData.columns:
            self.original_slide = list(self.slideData['Original Slide Path'])
        else:
            self.original_slide = [0] * len(self.files)
        self.positional_embeddings = []
        self.type_embed = type_embed
        self.mode = mode

    def __getitem__(self, index):
        label = self.labels[index]
        patient_id = None  
        try:
            slide = openslide.OpenSlide(self.original_slide[index])
            magnification = int(slide.properties.get("openslide.objective-power", 40)) 
        except Exception as e:
            magnification = 40  
        original_size = best_size(magnification, 256, 20)

        try:
            with h5py.File(self.files[index], 'r') as hdf5_file:
                patient_id = os.path.basename(self.files[index]).replace(".h5", "")
                features = torch.from_numpy(hdf5_file['features'][:])
                if "coords" in hdf5_file.keys():
                    x , y = zip(*hdf5_file["coords"][:])
                else:
                    x = hdf5_file['x'][:]
                    y = hdf5_file['y'][:]
                if "tile_path" in hdf5_file.keys():
                    tile_paths = [path.decode('utf-8') for path in hdf5_file['tile_path'][:]]
                elif "tile_paths"in hdf5_file.keys():
                    tile_paths = [path.decode('utf-8') for path in hdf5_file['tile_paths'][:]]
                else:
                    tile_paths = [0] * len(x)
         
                if features.ndim == 3 and self.mode == "train" :
                    random_indices = np.random.randint(0, features.shape[1], size=features.shape[0])
                    features = features[np.arange(features.shape[0]), random_indices]
                elif features.ndim == 3 and self.mode != "train":
                    features = features[np.arange(features.shape[0]), np.zeros(features.shape[0], dtype = int)]
                else:
                    features = features
                scales = 64  
                magnifications = hdf5_file.get('mag', np.array([40]))[0]  
                if self.type_embed == "2D":
                    
                    positional_embed = torch.from_numpy(np.array([
                        positional_embeddings_sin_cos(x_coord, y_coord) for x_coord, y_coord in zip(x, y)
                    ]))
                elif self.type_embed == "local":
                    k = 6
                    coords = [(x_, y_) for x_,y_ in zip(np.array(x), np.array(y))]
                    coords = np.array(coords)
                    knn_indices = compute_knn(coords, k)
                    local_embeddings = aggregate_local_features_euclidean(np.array(features), coords, knn_indices)
                    positional_embed = torch.tensor(local_embeddings)
                else:
                    positional_embed = torch.empty(2048)
                    

            return features, positional_embed, label, torch.tensor(x), torch.tensor(y), tile_paths, scales, original_size, patient_id
        except Exception as e:
            print(f"Warning: Failed to load HDF5 file for index {index}: {e}")
            return (torch.empty(0), torch.empty(0), label, [], [], [], 64, original_size, "error")
    def __len__(self):
        return (len(self.samples))
    def get_labels(self):
        return self.labels
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
