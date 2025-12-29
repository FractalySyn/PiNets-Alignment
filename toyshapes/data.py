import numpy as np
import pandas as pd
import scipy.ndimage
import torch









def make_dataset(n_blocks, N, target_shape, shapes, rel_freq, **kwargs):
  """  Synthetic Dataset Generator (does not generate images).
    Args: 
      n_blocks (int): Number of blocks in each image (must be a perfect square).
      N (int): Number of images to generate.
      target_shape (str): The positive class in binary classification — the shape to identify.
      shapes (list): List of possible shapes among ('square', 'triangle', 'circle', 'void'). 
      rel_freq (float): Relative frequency of the target shape in the dataset.
    Returns:
      data (pd.DataFrame): DataFrame containing the generated dataset, i.e. what shapes are in each block and the target variable.
  """

  assert int(n_blocks ** 0.5) ** 2 == n_blocks, "n_blocks must be a perfect square"
  assert target_shape in shapes, "target_shape must be one of the shapes"

  ## compute probabilities
  rel_freqs = np.ones(len(shapes)) * rel_freq
  rel_freqs[np.where(np.array(shapes) == target_shape)[0][0]] = 1
  probs = rel_freqs / np.sum(rel_freqs)

  ## generate dataset
  values = np.random.choice(shapes, size=(N, n_blocks), p=probs)
  y = (values == target_shape).sum(axis=1)
  data = pd.DataFrame(values, columns=[f'block_{i}' for i in range(n_blocks)])
  data['y'] = y

  return data





def generate_images(seed, data, n_blocks=16, dim_per_block=20, target_shape='triangle', backgrounds=True, nonconvex=0.25, **kwargs):
  """  Generates images and masks from the dataset.
    Args:
      seed (int): Random seed for reproducibility.
      data (pd.DataFrame): Data containing the shapes for each image.
      n_blocks (int): Number of blocks in each image (must be a perfect square).
      dim_per_block (int): Dimension of each block in pixels.
      target_shape (str): The positive class in binary classification — the shape to identify.
      backgrounds (bool): If True, uses random backgrounds; otherwise, use black background.
      nonconvex (float): Probability of generating non-convex shapes.
    Returns:
      images (array): Array of generated images.
      masks (array): Array of generated masks (ground-truth detection).
  """

  assert nonconvex >= 0.0 and nonconvex <= 1.0, "nonconvex must be in [0, 1]"

  ## drop duplicates if the generator has a high degree of homogeneity
  if not backgrounds and nonconvex == 0.0:
    data.drop_duplicates(inplace=True)

  ## generate images and masks (ground-truth detection)
  seeds = np.arange(seed, seed + data.shape[0])
  images, masks = [], []
  for row, seed_ in zip(data.iloc[:, :-1].values, seeds):
    im, mask = generate_image_array(row, n_blocks, dim_per_block, target_shape, seed_, backgrounds, nonconvex)
    images.append(im); masks.append(mask)
  
  return np.array(images), np.array(masks)



def generate_image_array(row, n_blocks, dim_per_block, target_shape, seed, backgrounds, nonconvex, **kwargs):
    
    sqrt_blocks = int(n_blocks ** 0.5)
    assert sqrt_blocks ** 2 == n_blocks, "n_blocks must be a perfect square"

    np.random.seed(seed)
    background = np.random.randint(50, 200) if backgrounds else 0
    shape_scales = np.random.uniform(0.5, 0.8, size=n_blocks)

    im_size = (sqrt_blocks * dim_per_block, sqrt_blocks * dim_per_block)
    image = np.zeros(im_size, dtype=np.uint8)
    mask = np.zeros(im_size, dtype=np.uint8)

    for i, shape in enumerate(row):
        x = (i % sqrt_blocks) * dim_per_block
        y = im_size[1] - ((i // sqrt_blocks) + 1) * dim_per_block
        scale = shape_scales[i]
        block, block_m = generate_shape_block(
            shape, dim_per_block, scale, background, nonconvex=nonconvex, target_shape=target_shape
        )

        image[y:y + dim_per_block, x:x + dim_per_block] = block
        mask[y:y + dim_per_block, x:x + dim_per_block] = block_m

    return image/255, mask




def generate_shape_block(shape, dim, scale, background, nonconvex, target_shape, **kwargs):
    """
    Generates a block with a given shape, possibly non-convex (hollow), and shades.
      Args:
        shape (str): The shape to generate ('square', 'circle', 'triangle', 'void').
        dim (int): Dimension of the block in pixels.
        scale (float): Scale factor for the size of the shape.
        background (int): Background color value (0-255).
        nonconvex (float): Probability of generating a non-convex shape.
        target_shape (str): The target shape for binary classification ('square', 'circle', 'triangle').
      Returns:
        block (np.ndarray): The generated shape block as a 2D numpy array.
        mask (np.ndarray): The mask for the shape, where 1 indicates the presence of the target shape.
    """

    ## preliminaries
    block = np.ones((dim, dim), dtype=np.uint8) * background
    mask = np.zeros((dim, dim), dtype=np.uint8)
    scaled_size = int(dim * scale)
    offset = (dim - scaled_size) // 2

    # Decide convex/non-convex
    is_nonconvex = np.random.rand() < nonconvex
    thickness = max(1, scaled_size // 5) if is_nonconvex else 0

    # Shades
    val_candidates = np.linspace(0, 256)
    val_candidates = val_candidates[(val_candidates < background-50) | (val_candidates > background+50)]
    shape_shade = np.random.choice(val_candidates, size=1, replace=False)[0]

    if shape == 'square':
        block[offset:offset + scaled_size, offset:offset + scaled_size] = shape_shade
        if target_shape == 'square':
            mask[offset:offset + scaled_size, offset:offset + scaled_size] = 1
        if is_nonconvex:
            inner_offset = offset + thickness
            inner_size = scaled_size - 2 * thickness
            if inner_size > 0:
                block[inner_offset:inner_offset + inner_size, inner_offset:inner_offset + inner_size] = background
                if target_shape == 'square':
                    mask[inner_offset:inner_offset + inner_size, inner_offset:inner_offset + inner_size] = 0

    elif shape == 'circle':
        center = dim // 2
        radius_outer = scaled_size // 2
        radius_inner = radius_outer - thickness if is_nonconvex else 0
        for r in range(dim):
            for c in range(dim):
                dist = ((r - center) ** 2 + (c - center) ** 2) ** 0.5
                if is_nonconvex:
                    if radius_inner < dist < radius_outer:
                        block[r, c] = shape_shade
                        if target_shape == 'circle':
                            mask[r, c] = 1
                else:
                    if 0 <= dist < int(radius_outer):
                        block[r, c] = shape_shade
                        if target_shape == 'circle':
                            mask[r, c] = 1

    elif shape == 'triangle':
        filling_speed = np.random.uniform(1-scale, 1-scale+.2)
        for row_t in range(scaled_size):
            row_idx = offset + row_t
            half_width = int(row_t * filling_speed)
            start_col = offset + (scaled_size // 2) - half_width
            end_col = offset + (scaled_size // 2) + half_width + 1
            if 0 <= row_idx < dim:
                start_col = max(start_col, 0)
                end_col = min(end_col, dim)
                if start_col < end_col:
                    block[row_idx, start_col:end_col] = shape_shade
                    if target_shape == 'triangle':
                        mask[row_idx, start_col:end_col] = 1
        if is_nonconvex and thickness > 0:
            for row_t in range(thickness, scaled_size-thickness):
                row_idx = offset + row_t
                half_width = int(row_t * filling_speed)
                start_col = offset + (scaled_size // 2) - half_width + thickness
                end_col = offset + (scaled_size // 2) + half_width - thickness + 1
                if 0 <= row_idx < dim:
                    start_col = max(start_col, 0)
                    end_col = min(end_col, dim)
                    if start_col < end_col:
                        block[row_idx, start_col:end_col] = background
                        if target_shape == 'triangle':
                            mask[row_idx, start_col:end_col] = 0

    else:  # void
        pass

    return block, mask











def split_data(images, masks, target, seed, val_size=0.2, verbose=False, **kwargs):
  """ Splits the dataset into train, validation, and test sets.
    Args:
      images (array): Array of images.
      masks (array): Array of masks (ground-truth detection).
      target (array): Array of target (binary).
      seed (int): Random seed for reproducibility.
      test_size (float): Proportion of the dataset to include in the test split.
      val_size (float): Proportion of the training set to include in the validation split.
    Returns:
      X_train, y_train, Pi_star_train: Training data, labels, and masks.
      X_val, y_val, Pi_star_val: Validation data, labels, and masks.
      X_test, y_test, Pi_star_test: Test data, labels, and masks.
  """
  
  X = torch.tensor(images, dtype=torch.float32)
  Pi_star = torch.tensor(masks, dtype=torch.float32)
  y = torch.tensor((target > 0) * 1, dtype=torch.float32)
  N = len(X)

  np.random.seed(seed)
  val_idx = np.random.choice(np.arange(N), size=int(N * val_size), replace=False)
  train_idx = np.setdiff1d(np.arange(N), val_idx)

  X_train, y_train, Pi_star_train = X[train_idx], y[train_idx], Pi_star[train_idx]
  X_val, y_val, Pi_star_val = X[val_idx], y[val_idx], Pi_star[val_idx]

  if verbose:
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

  return X_train, y_train, Pi_star_train, X_val, y_val, Pi_star_val