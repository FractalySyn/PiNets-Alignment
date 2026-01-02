import torch
import torch.nn as nn
import torch.nn.functional as F


from floods.utils import save_model_except_encoder, save_full_model



class MAE(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, pred, true):
    delta = (true - pred).abs()
    return delta.mean()
    
class FAR(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, pi, pi_star):
    FAR = 0.0
    for k in range(pi.shape[1]):
      pi_k = pi[:, k, :, :]
      pi_star_k = (pi_star == k-1.0).float()
      tdr = (pi_k * pi_star_k).sum(dim=(1,2)) / (pi_star_k.sum(dim=(1,2)) + 1e-6)
      FAR += (1 - tdr).mean()
    return FAR / pi.shape[1]
  
class IoU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, pi, pi_star):
    iou = 0.0
    for k in range(pi.shape[1]):
      pi_k = (pi.argmax(dim=1) == k).float()
      pi_star_k = (pi_star == k-1.0).float()
      intersection = (pi_k * pi_star_k).sum(dim=(1, 2))
      union = (pi_k + pi_star_k).sum(dim=(1, 2)) - intersection
      iou += (intersection / (union + 1e-6)).mean()
    return iou / pi.shape[1]
  
class BCE(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, pi, pi_star):
    total_loss = 0.0
    for k in range(pi.shape[1]):
      pi_k = pi[:, k, :, :]
      pi_star_k = (pi_star == k-1.0).float()
      loss = F.binary_cross_entropy(pi_k, pi_star_k, reduction='none')
      total_loss += loss.mean()
    return total_loss / pi.shape[1]
  












def training(model, train_loader, val_loader, optimizer, scheduler, loss_, n_epochs, device, ES, path):

  if loss_ == 'MAE':
    Loss = MAE()
  elif loss_ == 'FAR':
    Loss = FAR()
  elif loss_ == 'BCE':
    Loss = BCE()
  else:
    raise NotImplementedError(f'Loss {loss_} not implemented! Choose from MAE, FAR, BCE.')

  best_score = 1e6 if loss_ == 'MAE' else 0.0
  for epoch in range(n_epochs):
    model.train()

    if optimizer.param_groups[0]['lr'] <= ES:
      return model
    
    train_loss = 0.0
    for x, y, pi_star in train_loader:
      x, y, pi_star = x.to(device), y.to(device), pi_star.to(device)
      pi, y_pred = model(x)
      loss = Loss(y_pred, y) if loss_ == 'MAE' else Loss(pi, pi_star)
      optimizer.zero_grad(); loss.backward(); optimizer.step()
      train_loss += loss.item() * x.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_mae = 0.0; val_far = 0.0; val_iou = 0.0
    with torch.no_grad():
      for x, y, pi_star in val_loader:
        x, y, pi_star = x.to(device), y.to(device), pi_star.to(device)
        pi, y_pred = model(x)
        val_mae += MAE()(y_pred, y).item() * x.size(0)
        val_far += FAR()(pi, pi_star).item() * x.size(0)
        val_iou += IoU()(pi, pi_star).item() * x.size(0)
      val_mae /= len(val_loader.dataset)
      val_far /= len(val_loader.dataset)
      val_iou /= len(val_loader.dataset)

    ## A PiNet (trained on image-level labels) must be validated based on image-level performance, here MAE, as pixel-level info is assumed unavailable
    if loss_ == 'MAE':
      scheduler.step(val_mae)
      if val_mae < best_score:
        best_score = val_mae
        save_model_except_encoder(model, path) if model.load_pretrained else save_full_model(model, path)
    ## The SegNet (trained on pixel-level labels) is validated based on its TDR (1 - FAR), i.e. true positives
    else:
      scheduler.step(val_far)
      if 1 - val_far > best_score:
        best_score = 1 - val_far
        save_model_except_encoder(model, path) if model.load_pretrained else save_full_model(model, path)
                      
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.3f},",
          f"Val MAE: {val_mae:.0f}, Val TDR: {1-val_far:.3f}",
          f"Val IoU: {val_iou:.3f}",
          f"--- lr: {optimizer.param_groups[0]['lr']:.1e}")
    
  return model







def training_segbase(model, train_loader, val_loader, optimizer, scheduler, loss_, n_epochs, device, ES, path):

  if loss_ == 'FAR':
    Loss = FAR()
  elif loss_ == 'BCE':
    Loss = BCE()
  else:
    raise NotImplementedError(f'Loss {loss_} not implemented! Choose from FAR, BCE.')

  best_tdr = 0.0
  for epoch in range(n_epochs):
    model.train()

    if optimizer.param_groups[0]['lr'] <= ES:
      return model
    
    train_loss = 0.0
    for x, y, pi_star in train_loader:
      x, y, pi_star = x.to(device), y.to(device), pi_star.to(device)
      pi, y_pred = model(x)
      loss = Loss(pi, pi_star)
      optimizer.zero_grad(); loss.backward(); optimizer.step()
      train_loss += loss.item() * x.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_far = 0.0; val_iou = 0.0
    with torch.no_grad():
      for x, y, pi_star in val_loader:
        x, y, pi_star = x.to(device), y.to(device), pi_star.to(device)
        pi, y_pred = model(x)
        val_far += FAR()(pi, pi_star).item() * x.size(0)
        val_iou += IoU()(pi, pi_star).item() * x.size(0)
      val_far /= len(val_loader.dataset)
      val_iou /= len(val_loader.dataset)
    scheduler.step(val_far)

    if 1 - val_far > best_tdr:
      best_tdr = 1 - val_far
      save_model_except_encoder(model, path)
                      
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.3f},",
          f"Val TDR: {1-val_far:.3f}",
          f"Val IoU: {val_iou:.3f}",
          f"--- lr: {optimizer.param_groups[0]['lr']:.1e}")
    
  return model



def training_pinet(model, train_loader, weak_loader, gamma, val_loader, optimizer, scheduler, n_epochs, device, ES, path):

  best_tdr = 0.0
  for epoch in range(n_epochs):
    model.train()

    if optimizer.param_groups[0]['lr'] <= ES:
      return model
    
    train_loss = 0.0
    for (x, y, pi_star), (xw, yw, z) in zip(train_loader, weak_loader):
      x, y, pi_star = x.to(device), y.to(device), pi_star.to(device)
      xw, yw, z = xw.to(device), yw.to(device), z.to(device)
      pi, _ = model(x, z=None) 
      _, y_predw = model(xw, z)
      loss = gamma * BCE()(pi, pi_star) + SDLoss()(y_predw, yw)
      optimizer.zero_grad(); loss.backward(); optimizer.step()
      train_loss += loss.item() * x.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_far = 0.0; val_iou = 0.0
    with torch.no_grad():
      for x, y, pi_star in val_loader:
        x, y, pi_star = x.to(device), y.to(device), pi_star.to(device)
        pi, y_pred = model(x, z=None)
        val_far += FAR()(pi, pi_star).item() * x.size(0)
        val_iou += IoU()(pi, pi_star).item() * x.size(0)
      val_far /= len(val_loader.dataset)
      val_iou /= len(val_loader.dataset)
    scheduler.step(val_far)

    if 1 - val_far > best_tdr:
      best_tdr = 1 - val_far
      save_model_except_encoder(model, path)
                      
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.3f},",
          f"Val TDR: {1-val_far:.3f}",
          f"Val IoU: {val_iou:.3f}",
          f"--- lr: {optimizer.param_groups[0]['lr']:.1e}")
    
  return model










def eval(model, test_loader, device, pi=None):

  if pi is None:
    model.eval(); pi = []; y_pred = []
    with torch.no_grad():
      for x, y, pi_star in test_loader:
        pi_, y_ = model(x.to(device))
        pi.append(pi_); y_pred.append(y_)

    pi = torch.cat(pi, dim=0)
    pic = pi.argmax(dim=1)
    y = test_loader.dataset.tensors[1].to(device)
    y_pred = torch.cat(y_pred, dim=0)
    mae = (y_pred[:, 2] - y[:, 2]).abs().mean().item()
  else:
    y = test_loader.dataset.tensors[1].to(device)
    pic = pi+1
    y_pred = (pic==2).float().sum(dim=(1,2))
    mae = (y_pred - y[:, 2]).abs().mean().item()

  pi_star = (test_loader.dataset.tensors[2] == 1).float().to(device)
  pi_water = (pic==2).float()
  tdr_water = (pi_water * pi_star).sum() / pi_star.sum()

  intersection = (pi_water * pi_star).sum(dim=(1, 2))
  union = (pi_water + pi_star).sum(dim=(1, 2)) - intersection
  iou_water = (intersection / (union + 1e-6)).mean().item()

  pi_star = (test_loader.dataset.tensors[2] == 0).float().to(device)
  pi_nowater = (pic==1).float()
  tdr_nowater = (pi_nowater * pi_star).sum() / pi_star.sum()

  intersection = (pi_nowater * pi_star).sum(dim=(1, 2))
  union = (pi_nowater + pi_star).sum(dim=(1, 2)) - intersection
  iou_nowater = (intersection / (union + 1e-6)).mean().item()

  print(f'Test MAE: {mae:.0f},', 
        f'Test IoU water: {iou_water:.3f},',
        f'Test IoU no water: {iou_nowater:.3f},',
        f'Test TDR water: {tdr_water:.3f}',
        f'Test TDR no water: {tdr_nowater:.3f}')
  
  return pic.cpu()






class SDLoss(nn.Module):
  """ Entropy (NLL) and optional L2 regularization on the detection map. 
    Init:
      lam (float): Regularization coefficient for the L2 penalty.
    Forward:
      logits (torch.Tensor): Predicted logits.
      y_true (torch.Tensor): True labels.
      pi (torch.Tensor, optional): Detection map.
  """

  def __init__(self, lam=0., **kwargs):
    super().__init__()
    self.lam = lam

  def forward(self, logits, y_true, pi=None):
    entropy = F.binary_cross_entropy_with_logits(logits, y_true, reduction='mean')
    if self.lam > 0 and pi is not None:
      penalty = (pi**2).sum(dim=(1, 2, 3)).mean() * self.lam
      return entropy + penalty
    else:
      return entropy
    