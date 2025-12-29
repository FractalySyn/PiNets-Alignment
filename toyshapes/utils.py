import os
import torch
import random
import shutil
import numpy as np



def env_reproducibility():
  """
  Set environment variables to ensure reproducibility.
  """
  os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.use_deterministic_algorithms(True, warn_only=True)

def seed_all(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)








def model_size(model):
  """ Calculate the size of a PyTorch model in parameters. """
  size = sum([len(p.flatten()) for p in model.parameters()])
  if size > 1e5:
    print(f'Model size: {size/1e6:.3f}M parameters')
  else:
    print(f'Model size: {size} parameters')



def create_dir(path, empty=False):
  """ Create a directory if it does not exist and optionally empties it. """
  if empty:
    shutil.rmtree(path, ignore_errors=True)
  if not os.path.exists(path):
    os.makedirs(path)
  return path









def make_legend_label(model_name):
  pieces = []
  if "pinets" in model_name:
    pieces.append("PiNet")
    pieces.append("Soft") if "soft" in model_name else None
    pieces.append('Naive')  if 'naive' in model_name else None
    pieces.append("Feedback") if "feedback" in model_name else None
    pieces.append("Ensemble") if "ensemble" in model_name else None
    pieces.append("Strong") if "strong" in model_name else None
  else: 
    pieces.append("Vanilla Grad") if "vanilla" in model_name else None
    pieces.append("Grad-CAM") if "gradcam" in model_name else None
  return ' '.join(pieces)












def compute_ellipse_parameters(x, y):
    """
    Compute the parameters for an ellipse that fits a set of points (x, y).

    Parameters:
        x (array-like): X-coordinates of the points.
        y (array-like): Y-coordinates of the points.

    Returns:
        dict: A dictionary containing:
            - 'mean': Center of the ellipse (mean of x and y).
            - 'cov': Covariance matrix of the points.
            - 'eigenvalues': Eigenvalues of the covariance matrix.
            - 'eigenvectors': Eigenvectors of the covariance matrix.
            - 'angle': Rotation angle of the ellipse in degrees.
            - 'width': Width of the ellipse (2 * sqrt(largest eigenvalue)).
            - 'height': Height of the ellipse (2 * sqrt(smallest eigenvalue)).
    """
    # Combine x and y into a 2D array
    points = np.column_stack((x, y))
    
    # Compute the mean of the points
    mean = points.mean(axis=0)
    
    # Compute the covariance matrix
    cov = np.cov(points, rowvar=False)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Compute the rotation angle in degrees
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    # Compute the width and height of the ellipse (2 standard deviations)
    width = 2.45 * np.sqrt(eigenvalues[0])
    height = 2.45 * np.sqrt(eigenvalues[1])
    
    return {
        'mean': mean,
        'cov': cov,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'angle': angle,
        'width': width,
        'height': height
    }








def inner_false_indices(arr):
    """
    Returns indices of False values in arr, excluding leading and trailing False runs.
    If all values are False, returns an empty array.
    """
    if not np.any(arr):
        return np.array([], dtype=int)
    first_true = np.argmax(arr)
    last_true = len(arr) - 1 - np.argmax(arr[::-1])
    return np.where((~arr) & (np.arange(len(arr)) > first_true) & (np.arange(len(arr)) < last_true))[0]