import torch
import torch.nn as nn
import torch.nn.functional as F









class CNN(nn.Module):
  """ A simple CNN for binary classification on images.
    Args:
      Cbase (int): Base number of channels for the first convolutional layer.
      depth (int): Number of residual blocks in the encoder.
      dropout (float): Dropout rate for regularization. Applied to each ResidualBlock.
      skip (bool): Whether to use skip connections in residual blocks.
    Returns:
      logits (torch.Tensor): 1D logits for binary classification.
      _: placeholder for unused output
      _: placeholder for unused output
  """

  def __init__(self, Cbase, depth, dropout=0.1, skip=True, **kwargs):
    super().__init__()
    self.encoder = Encoder(1, Cbase, depth, dropout, skip)
    self.fc = nn.Linear(Cbase*2**(depth-1), 1)

  def forward(self, x):
    x = x.unsqueeze(1) if x.ndim == 3 else x
    # h = self.encoder(x)
    feature_maps = self.encoder(x)
    h = F.adaptive_avg_pool2d(feature_maps, (1, 1)).squeeze()
    logits = self.fc(h).squeeze()
    return logits, feature_maps, None
  










class BinaryConvPiNet(nn.Module):
  """ Binary Convolutional PiNet for image classification and detection. Special case Z=X.
    Args:
      Cbase (int): Base number of channels for the first convolutional layer.
      depth (int): Number of residual blocks in the encoder and decoder.
      activation (callable): Activation function to apply to the output of the decoder, produces Pi(x).
      dropout (float): Dropout rate for regularization. Applied to each ResidualBlock.
      hardSL (bool): Whether to use a hard second-look (imposes to look at values z s.t. y=Sum[Pi(x)*z]) or a soft second-look (only imposes to look at space Z s.t. y=Sum[Pi(x)]).
      skip (bool): Whether to use skip connections in residual blocks.
      use_decoder (bool): Whether to use a decoder for the detection map Pi(x). If False, uses an average pooling followed by a linear layer.
    Returns:
      logits (torch.Tensor): 1D logits 
      pi (torch.Tensor): detection map Pi(x)
      piz (torch.Tensor): detected signal Pi(x)*z
  """

  def __init__(self, Cbase, depth, activation, dropout=0.0, hardSL=True, skip=False, use_decoder=True, zdim=None, **kwargs):

    super().__init__()
    self.hardSL = hardSL; self.use_decoder = use_decoder
    self.encoder = Encoder(1, Cbase, depth, dropout, skip)
    self.decoder = Decoder(1, Cbase, depth, dropout, skip) if use_decoder else nn.Linear(Cbase*2**(depth-1), zdim)
    self.activ = activation
    self.a, self.b = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.ones(1))

  def forward(self, x):

    x = x.unsqueeze(1) if x.ndim == 3 else x
    z = x.clone() # Z=X, only there for notational rigor

    ## Detector: encoder + decoder
    h = self.encoder(x)
    if self.use_decoder:
      pi = self.decoder(h).reshape(z.size())
    else: 
      h = F.adaptive_avg_pool2d(h, (1, 1)).squeeze()
      pi = self.decoder(h).reshape(z.size())

    pi = self.activ(pi)

    ## Second-look
    piz = pi * z
    if self.hardSL: 
      logits = piz.sum(dim=(1, 2, 3)) * self.b**2 + self.a
    else:
      logits = pi.sum(dim=(1, 2, 3)) * self.b**2 + self.a  

    return logits, pi, piz
  
    












class Encoder(nn.Module):
  """ Residual Encoder 
    Args:
      in_channels (int): Number of input channels.
      base_channels (int): Base number of channels for the first convolutional layer.
      depth (int): Number of residual blocks and downscaling.
      dropout (float): Dropout rate for regularization. Applied to each ResidualBlock.
      skip (bool): Whether to use skip connections in residual blocks.
    Returns:
      x (torch.Tensor): Output channels: base_channels*2^depth.
  """

  def __init__(self, in_channels=1, base_channels=32, depth=3, dropout=0.1, skip=True):

    super().__init__()
    self.depth = depth; self.skip = skip
    self.blocks = nn.ModuleList(); self.pools = nn.ModuleList()

    for i in range(depth):
      input_channels = in_channels if i == 0 else base_channels * (2 ** (i - 1))
      output_channels = base_channels * (2 ** i)
      self.blocks.append(ResidualBlock(input_channels, output_channels, dropout))
      self.pools.append(nn.MaxPool2d(2) if i < depth - 1 else nn.Identity())

  def forward(self, x):
    
    for i in range(self.depth):
      x = self.blocks[i](x, self.skip)
      x = self.pools[i](x)
    
    return x











class Decoder(nn.Module):
  """ Residual Decoder
    Args:
      out_channels (int): Number of output channels.
      base_channels (int): Base number of channels as in the encoder, used for symmetry.
      depth (int): Number of residual blocks and upscaling.
      dropout (float): Dropout rate for regularization. Applied to each ResidualBlock.
      skip (bool): Whether to use skip connections in residual blocks.
    Returns:
      x (torch.Tensor): Output channels: out_channels.
  """

  def __init__(self, out_channels=1, base_channels=32, depth=3, dropout=0.1, skip=True):

    super().__init__()
    self.depth = depth; self.skip = skip
    self.upsamples = nn.ModuleList(); self.blocks = nn.ModuleList()

    for i in range(depth - 1, 0, -1):
      input_channels = base_channels * (2 ** i)
      output_channels = base_channels * (2 ** (i - 1))
      self.upsamples.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2))
      self.blocks.append(ResidualBlock(output_channels, output_channels, dropout))
    self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

  def forward(self, x):

    for i in range(self.depth - 1):
      x = self.upsamples[i](x)
      x = self.blocks[i](x, self.skip)

    return self.final(x)








class ResidualBlock(nn.Module):
  """ Residual Block with two convolutional layers, batch normalization, ReLU activation, and optional skip connection.
    Args:
      in_channels (int): Number of input channels.
      out_channels (int): Number of output channels.
      dropout (float): Dropout rate for regularization.
    Returns:
      x (torch.Tensor): Output channels
  """

  def __init__(self, in_channels, out_channels, dropout=0.1):

    super().__init__()
    self.same_shape = in_channels == out_channels
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, 3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.Dropout(dropout)
    )
    self.skip = nn.Identity() if self.same_shape else nn.Conv2d(in_channels, out_channels, kernel_size=1)
    self.activation = nn.ReLU()

  def forward(self, x, skip=False):
    
    if skip:
      return self.activation(self.conv(x) + self.skip(x))
    else:
      return self.activation(self.conv(x)) 