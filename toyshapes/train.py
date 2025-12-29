import torch
import torch.nn as nn
import torch.nn.functional as F

from toyshapes.eval import Accuracy




def training_loop(model, optim, train_loader, val_loader, Loss, n_epochs, device, seed, verbose=False,
                  min_acc=0.5, alpha_stab=0, gamma=0, freq_eval=10, path='', **kwargs):
  """ 
  Trains a model with a given optimizer and loss function for n_epochs.
  Evaluates on validation set every freq_eval steps. 
  Early stops at Acc=100%.
  Saves if accuracy exceeds min_acc.
    Args:
      model (torch.nn.Module): The model to train.
      optim (torch.optim.Optimizer): The optimizer for training.
      train_loader (torch.utils.data.DataLoader): DataLoader for training data.
      val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
      Loss (callable): Loss function.
      n_epochs (int): Max number of epochs.
      device (torch.device): Device.
      seed (int): Random seed.
      verbose (bool): Whether to print training progress.
      min_acc (float): Minimum accuracy to save the model.
      alpha_stab (float): Regularization parameter for PiNet.
      freq_eval (int): Frequency of evaluation on validation set and saving.
      path (str): Folder where to save the model.
    Returns:
    bool: True if training was successful (Accuracy >= min_acc), False otherwise.
    Note: let alpha params (feedback) to 0 if the model is not a PiNet
  """

  best_acc = 0
  for epoch in range(n_epochs):
    for step, (x, y, pi_star) in enumerate(train_loader):

      ## training step and prediction loss
      model.train()
      x, y = x.to(device), y.to(device)
      pred, pi, piz = model(x)
      loss = Loss(pred, y, pi)

      ## PiNet augmentation
      if gamma > 0:
        pi_star = pi_star.to(device) 
        idx = (y == 1).nonzero(as_tuple=False)[0]
        detection_loss = ((pi_star[idx].squeeze() - pi[idx].squeeze())**2).mean() * gamma
        loss += detection_loss

      ## feedback loss for PiNet
      if alpha_stab > 0:
        _, pi_, _ = model(F.relu(piz))
        loss += ((pi - pi_)**2).mean() * alpha_stab

      optim.zero_grad(); loss.backward(); optim.step()

      ## evaluation and saving
      if (step+1) % freq_eval == 0:

        model.eval()
        with torch.no_grad():
          preds = [model(x.to(device))[0] for (x, _) in val_loader]
        preds = F.sigmoid(torch.cat(preds)) > 0.5
        acc = Accuracy(val_loader.dataset.tensors[1], preds.float().detach().cpu())
        model.train()

        if acc > best_acc:
          best_acc = acc
          if acc >= min_acc:
            torch.save(model.state_dict(), path + f'model_{seed}.pth')
          if verbose:
            print(f"Epoch {epoch+1}/{n_epochs} step {step+1}/{len(train_loader)}\t" +\
                f"---\t Loss: {loss.item():.4f}  Accuracy: {acc*100:.2f}% ")
          
      ## early stopping
      if best_acc == 1:
        return best_acc
  return True if best_acc >= min_acc else False







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
    return entropy
    