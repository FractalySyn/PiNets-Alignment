import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import gc
import rasterio
import matplotlib.pyplot as plt




def load_X(row, DIR):
  """ Load Sentinel-2 data given a row and directory. """
  s2_path = DIR+'S2Hand/' + row[0].replace('S1', 'S2')
  with rasterio.open(s2_path) as src:
    return src.read()[[1,2,3,8,11,12]] 
  
def load_pi(row, DIR):
  """ Load ground truth masks given a row and directory. """
  mask_path = DIR+'LabelHand/' + row[1]
  with rasterio.open(mask_path) as src:
    return src.read(1) 
  







def split_into_insets(X):
    """ Splits 512x512 images into four 256x256 insets.
    Args:
        X (np.ndarray): Input data of shape (N, C, 512, 512) or (N, 512, 512).
    Returns:
        np.ndarray: Insets of shape (N, 4, C, 256, 256) or (N, 4, 256, 256)."""
    
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
  """ Merges four 256x256 insets back into a single 512x512 image.
  Args:
      insets (np.ndarray): Insets of shape (N, 4, C, 256, 256) or (N, 4, 256, 256).
  Returns:
      np.ndarray: Merged images of shape (N, C, 512, 512) or (N, 512, 512).
  """

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




def flatten_data(X_insets, pi_star_insets, y_insets):
    """ Flattens the insets of the input data, ground truth masks, and labels into 2D tensors.
    Args:
        X_insets (np.ndarray): Input data insets of shape (N, num_insets, C, H, W).
        pi_star_insets (np.ndarray): Ground truth mask insets of shape (N, num_insets, H, W).
        y_insets (np.ndarray): Labels insets of shape (N, num_insets) or (N, num_insets, num_classes).
    Returns:
        tuple: Flattened tensors (X_flat, pi_star_flat, y_flat).
    """

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
    """ Splits the flattened data into training and validation sets based on the specified validation split ratio.
    Args:
        X_flat (np.ndarray): Flattened input data of shape (N, C, H, W).
        pi_star_flat (np.ndarray): Flattened ground truth masks of shape (N, H, W).
        area_flat (np.ndarray): Flattened labels of shape (N, ) or (N, num_classes).
        val_split (float): Proportion of the data to be used for validation (between 0 and 1).
    Returns:
        tuple: Training and validation sets (X_train, X_val, pi_star_train, pi_star_val, area_train, area_val).
    """

    val_idx = np.random.choice(len(X_flat), int(len(X_flat) * val_split), replace=False)
    train_idx = np.array([i for i in range(len(X_flat)) if i not in val_idx])
    
    X_train, X_val = X_flat[train_idx], X_flat[val_idx]
    pi_star_train, pi_star_val = pi_star_flat[train_idx], pi_star_flat[val_idx]
    area_train, area_val = area_flat[train_idx], area_flat[val_idx]
    
    return X_train, X_val, pi_star_train, pi_star_val, area_train, area_val







def save_model_except_encoder(model, save_path):
    state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("encoder.")}
    torch.save(filtered_state_dict, save_path)

def save_full_model(model, save_path):
  state_dict = model.state_dict()
  torch.save(state_dict, save_path)





def visualize_predictions(pi1, pi2, pi_star, scenes, indices, names=('Predicted1', 'Predicted2'), cmap=None, norm=None):
    """ Visualizes Sentinel-2 scenes, ground truth masks, and two sets of predicted masks side by side.
    Args:
        pi1 (np.ndarray): First set of predicted masks of shape (N, H, W).
        pi2 (np.ndarray): Second set of predicted masks of shape (N, H, W).
        pi_star (np.ndarray): Ground truth masks of shape (N, H, W).
        scenes (np.ndarray): Sentinel-2 scenes of shape (N, H, W, C).
        indices (list): List of indices to visualize.
        names (tuple): Names for the two predicted masks.
        cmap (matplotlib.colors.Colormap): Colormap for mask visualization.
        norm (matplotlib.colors.Normalize): Normalization for mask visualization.
    Saves the visualization as 'floods/floods.png' and displays it.
    """
    
    n = len(indices)
    fig, axes = plt.subplots(4, n, figsize=(2.5 * n, 10), dpi=200, gridspec_kw={'wspace':0.05, 'hspace':0.05})

    for i, idx in enumerate(indices):
        # Sentinel-2 scene (ensure valid RGB range for float images)
        scene = scenes[idx].copy()
        axes[0, i].imshow(np.clip(scene, 0.0, 1.0))
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

    plt.savefig('floods/floods.png', dpi=200, bbox_inches='tight')
    plt.show()













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

