import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import gc
import rasterio
import matplotlib.pyplot as plt




def augment_data(X, pi, y):
    """
    Augments the dataset by applying horizontal and vertical flips to each sample.
    Ensures the order of augmented tensors is consistent across X, pi, and y.

    Args:
        X (torch.Tensor): Input tensor of shape (N, C, H, W).
        pi (torch.Tensor): Mask tensor of shape (N, H, W).
        y (torch.Tensor): Labels tensor of shape (N, ...).

    Returns:
        tuple: Augmented tensors (X_aug, pi_aug, y_aug).
    """
    N = X.shape[0]
    aug_X, aug_pi, aug_y = [], [], []

    for i in range(N):
        x, m, label = X[i], pi[i], y[i]
        aug_X.append(x); aug_pi.append(m); aug_y.append(label)
        # Horizontal flip
        x_h, m_h = torch.flip(x, dims=[2]), torch.flip(m, dims=[1])
        aug_X.append(x_h); aug_pi.append(m_h); aug_y.append(label)
        # Vertical flip
        x_v, m_v = torch.flip(x, dims=[1]), torch.flip(m, dims=[0])
        aug_X.append(x_v); aug_pi.append(m_v); aug_y.append(label)

    X_aug = torch.stack(aug_X, dim=0); pi_aug = torch.stack(aug_pi, dim=0); y_aug = torch.stack(aug_y, dim=0)
    return X_aug, pi_aug, y_aug


def collect_scenes(X, y, pi, n_scenes, seed, val_split=1/3):

  # sample
  seed_all(seed)
  idx = torch.randperm(X.shape[0])[:n_scenes]
  X = X[idx]; y = y[idx]; pi = pi[idx]

  # split
  if val_split == 0:
    X, pi, y = flatten_data(X, pi, y)
    X, pi, y = augment_data(X, pi, y)
    return X, pi, y
  
  n_val = 1 if n_scenes <= 3 else int(round(n_scenes * val_split))
  val_idx = np.random.choice(len(X), n_val, replace=False)
  X_val, pi_val, y_val = X[val_idx], pi[val_idx], y[val_idx]
  train_idx = np.array([i for i in range(len(X)) if i not in val_idx])
  X, pi, y = X[train_idx], pi[train_idx], y[train_idx]

  # flatten and augment
  X, pi, y = flatten_data(X, pi, y)
  X, pi, y = augment_data(X, pi, y)
  X_val, pi_val, y_val = flatten_data(X_val, pi_val, y_val)
  X_val, pi_val, y_val = augment_data(X_val, pi_val, y_val)

  return X, pi, y, X_val, pi_val, y_val, idx










def flatten_data(X_insets, pi_star_insets, y_insets):

    _, _, C, H, W = X_insets.shape
    X_flat = torch.tensor(X_insets.reshape(-1, C, H, W)).float()
    
    _, _, H, W = pi_star_insets.shape
    pi_star_flat = torch.tensor(pi_star_insets.reshape(-1, H, W)).float()
    
    if len(y_insets.shape) == 2:  # binary 
        y_flat = torch.tensor(y_insets.reshape(-1, 1)).float()
    else:   
        y_flat = torch.tensor(y_insets.reshape(-1, 3)).float()
    
    return X_flat, pi_star_flat, y_flat








def train_val_split(X_flat, pi_star_flat, area_flat, val_split):

    val_idx = np.random.choice(len(X_flat), int(len(X_flat) * val_split), replace=False)
    train_idx = np.array([i for i in range(len(X_flat)) if i not in val_idx])
    
    X_train, X_val = X_flat[train_idx], X_flat[val_idx]
    pi_star_train, pi_star_val = pi_star_flat[train_idx], pi_star_flat[val_idx]
    area_train, area_val = area_flat[train_idx], area_flat[val_idx]
    
    return X_train, X_val, pi_star_train, pi_star_val, area_train, area_val





def load_X(row):
  s2_path = 'floods/S2Hand/' + row[0].replace('S1', 'S2')
  with rasterio.open(s2_path) as src:
    return src.read()[[1,2,3,8,11,12]] 
  
def load_pi(row):
  mask_path = 'floods/LabelHand/' + row[1]
  with rasterio.open(mask_path) as src:
    return src.read(1) 





def split_into_insets(X):
    if X.ndim == 4:  # Input has a channel dimension
        N, C, H, W = X.shape
        assert H == 512 and W == 512, "Input dimensions must be 512x512."
        insets = np.zeros((N, 4, C, 256, 256), dtype=X.dtype)
        insets[:, 0] = X[:, :, :256, :256]  # Top-left
        insets[:, 1] = X[:, :, :256, 256:]  # Top-right
        insets[:, 2] = X[:, :, 256:, :256]  # Bottom-left
        insets[:, 3] = X[:, :, 256:, 256:]  # Bottom-right
    elif X.ndim == 3:  # Input has no channel dimension
        N, H, W = X.shape
        assert H == 512 and W == 512, "Input dimensions must be 512x512."
        insets = np.zeros((N, 4, 256, 256), dtype=X.dtype)
        insets[:, 0] = X[:, :256, :256]  # Top-left
        insets[:, 1] = X[:, :256, 256:]  # Top-right
        insets[:, 2] = X[:, 256:, :256]  # Bottom-left
        insets[:, 3] = X[:, 256:, 256:]  # Bottom-right
    else:
        raise ValueError("Input must have 3 or 4 dimensions.")
    return insets


def merge_insets(insets):
  if len(insets.shape) == 4:
    N, _, H, W = insets.shape
    merged = np.zeros((N, 512, 512))
    merged[:, :256, :256] = insets[:, 0]  # Top-left
    merged[:, :256, 256:] = insets[:, 1]  # Top-right
    merged[:, 256:, :256] = insets[:, 2]  # Bottom-left
    merged[:, 256:, 256:] = insets[:, 3]  # Bottom-right
  elif len(insets.shape) == 5:
    N, _, H, W, C = insets.shape
    merged = np.zeros((N, 512, 512, C))
    merged[:, :256, :256, :] = insets[:, 0]  # Top-left
    merged[:, :256, 256:, :] = insets[:, 1]  # Top-right
    merged[:, 256:, :256, :] = insets[:, 2]  # Bottom-left
    merged[:, 256:, 256:, :] = insets[:, 3]  # Bottom-right
  return merged









def crash_ram():
  """ Crash the RAM by allocating a large tensor. To force Kernel and GPU reset. """
  return torch.ones((1000, 1000, 1000, 1000))

def free_ram(elements=[]):
  """  Free up RAM by deleting specified elements and running garbage collection. """
  [trydel(el) for el in elements]
  torch.cuda.empty_cache()
  gc.collect()

def trydel(obj):
  try: del obj
  except: pass




def seed_all(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)





def env_reproducibility():
  """
  Set environment variables to ensure reproducibility.
  """
  os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.use_deterministic_algorithms(True, warn_only=True)
  
  
  
  
  
  
  
def model_size(model):
  """ Calculate the size of a PyTorch model in parameters. """
  size = sum([len(p.flatten()) for p in model.parameters()])
  if size > 1e5:
    print(f'Model size: {size/1e6:.3f}M parameters')
  else:
    print(f'Model size: {size} parameters')


def visualize_predictions(pi1, pi2, pi_star, scenes, indices, names=('Predicted1', 'Predicted2'), cmap=None, norm=None):
    
    n = len(indices)
    fig, axes = plt.subplots(4, n, figsize=(2.5 * n, 10), dpi=100, gridspec_kw={'wspace':0.05, 'hspace':0.05})

    for i, idx in enumerate(indices):
        # Sentinel-2 scene
        axes[0, i].imshow(scenes[idx], vmax=1)
        axes[0, i].axis('off')
        # Ground truth mask
        axes[1, i].imshow(pi_star[idx], cmap=cmap, norm=norm)
        axes[1, i].axis('off')
        # First predicted mask
        axes[2, i].imshow(pi1[idx], cmap=cmap, norm=norm)
        axes[2, i].axis('off')
        # Second predicted mask
        axes[3, i].imshow(pi2[idx], cmap=cmap, norm=norm)
        axes[3, i].axis('off')

    # Annotations for rows (types)
    axes[0, 0].annotate('Sentinel-2', xy=(-0.05, 0.5), xycoords='axes fraction', va='center', ha='right', fontsize=14, fontweight='bold', rotation=90)
    axes[1, 0].annotate('Ground Truth', xy=(-0.05, 0.5), xycoords='axes fraction', va='center', ha='right', fontsize=14, fontweight='bold', rotation=90)
    axes[2, 0].annotate(names[0], xy=(-0.05, 0.5), xycoords='axes fraction', va='center', ha='right', fontsize=14, fontweight='bold', rotation=90)
    axes[3, 0].annotate(names[1], xy=(-0.05, 0.5), xycoords='axes fraction', va='center', ha='right', fontsize=14, fontweight='bold', rotation=90)

    plt.savefig('floods.png', dpi=200, bbox_inches='tight')
    plt.show()



def save_model_except_encoder(model, save_path):
    state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("encoder.")}
    torch.save(filtered_state_dict, save_path)

def save_full_model(model, save_path):
  state_dict = model.state_dict()
  torch.save(state_dict, save_path)


