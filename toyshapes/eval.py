import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from toyshapes.models import BinaryConvPiNet, CNN

import matplotlib.pyplot as plt
import numpy as np

import json
import os
from glob import glob





def evaluate_cnns(thresholds, saved_models, cnn_kwargs, X_test, y_test, Pi_star_test, device):

  test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
  cnn = CNN(**cnn_kwargs).to(device)
  cnn.eval()

  Vanilla = []; CAMs = []
  for path in saved_models:
      
    cnn.load_state_dict(torch.load(path, map_location=device))
    y_hat, Vgrads, GradCAMs = predict_cnn(cnn, test_loader, device)
    Vgrads = (Vgrads / Vgrads.max()).unsqueeze(1).abs()
    GradCAMs = GradCAMs / GradCAMs.max()

    Vanilla.append(evaluate(y_hat, Vgrads, X_test, y_test, Pi_star_test, thresholds, cnn, device))
    CAMs.append(evaluate(y_hat, GradCAMs, X_test, y_test, Pi_star_test, thresholds, cnn, device))

  return {'Vanilla': Vanilla, 'CAMs': CAMs}









def evaluate_pinets(thresholds, saved_models, pinet_kwargs, X_test, y_test, Pi_star_test, device):

  test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
  pinet = BinaryConvPiNet(**pinet_kwargs).to(device)
  pinet.eval()

  Pinets = []
  for path in saved_models:
      
    pinet.load_state_dict(torch.load(path, map_location=device))
    y_hat, Pi = predict_pinet(pinet, test_loader, device)
    Pi = Pi / Pi.max()

    Pinets.append(evaluate(y_hat, Pi, X_test, y_test, Pi_star_test, thresholds, pinet, device))

  return Pinets






def evaluate_ensemble(thresholds, indiv_models, pinet_kwargs, X_test, y_test, Pi_star_test, device):

  test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
  pinet = BinaryConvPiNet(**pinet_kwargs).to(device)
  pinet.eval()

  logits = torch.zeros_like(y_test, dtype=torch.float32).to(device)
  logits_on_piz = torch.zeros_like(y_test, dtype=torch.float32).to(device)
  Pi = torch.zeros_like(X_test.unsqueeze(1), dtype=torch.float32).to(device)

  for path in indiv_models:
    pinet.load_state_dict(torch.load(path, map_location=device))
    l, P = predict_pinet(pinet, test_loader, device, return_logits=True)
    logits += l / len(indiv_models) 
    Pi += pinet.b**2 * P / len(indiv_models)

  Pi = Pi / Pi.max()
  y_hat = (F.sigmoid(logits) > 0.5).float()

  pi_loader = DataLoader(TensorDataset(Pi, X_test.to(device)), batch_size=32, shuffle=False)
  for path in indiv_models:
    pinet.load_state_dict(torch.load(path, map_location=device))
    l = predict_on_piz(pinet, pi_loader, device)
    logits_on_piz += l / len(indiv_models)

  y_hat_on_piz = (F.sigmoid(logits_on_piz) > 0.5).float()
  acc_on_piz = Accuracy(y_hat_on_piz.cpu(), y_test).item()

  results = evaluate(y_hat, Pi, X_test, y_test, Pi_star_test, thresholds, pinet, device, skip_shift=True)
  results['acc_on_piz'] = acc_on_piz; results['acc_shift'] = results['acc'] - acc_on_piz

  return results





def evaluate(y_hat, Pi, X_test, y_test, Pi_star_test, thresholds, model, device, skip_shift=False):
    
  if not skip_shift:
    pi_loader = DataLoader(TensorDataset(Pi, X_test.to(device)), batch_size=32, shuffle=False)
    y_hat_on_piz = predict_on_piz(model, pi_loader, device)
    
  acc = Accuracy(y_hat.cpu(), y_test).item()
  acc_on_piz = Accuracy(y_hat_on_piz.cpu(), y_test).item() if not skip_shift else None
  acc_shift = acc - acc_on_piz if not skip_shift else None

  tdrs, tars = compute_metrics(Pi, Pi_star_test, y_test, thresholds, device)
  scores = tdrs * tars
  pi_best = (Pi.cpu() > thresholds[np.argmax(scores)]).float()

  optimal = eval_detection(pi_best, Pi_star_test, y_hat.cpu(), y_test, verbose=False)
  naive = eval_detection(F.relu(Pi).cpu(), Pi_star_test, y_hat.cpu(), y_test, verbose=False)

  return {
    'acc': acc, 'acc_on_piz': acc_on_piz, 'acc_shift': acc_shift,
    'naive': naive, 'optimal': optimal, 'tdrs': tdrs, 'tars': tars, 'scores': scores
  }
  








def predict_cnn(cnn, test_loader, device):

  y_hat, Vgrads, GradCAMs = [], [], []
  cnn.eval()

  for (x, _) in test_loader:

    x = x.clone().to(device).requires_grad_(True)
    logits, feature_maps, _ = cnn(x)

    logits.sum().backward(retain_graph=True)
    Vgrads.append(x.grad.detach().cpu())
    
    feature_maps.retain_grad()
    logits.sum().backward()
    gradients = feature_maps.grad  
    activations = feature_maps.detach()  
    
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [batch_size, channels, 1, 1]
    grad_cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [batch_size, 1, height, width]
    grad_cam = F.relu(grad_cam)    
    if grad_cam.shape[2:] != x.shape[1:]:
        grad_cam = F.interpolate(grad_cam, size=x.shape[1:], mode='bilinear', align_corners=False)
    
    GradCAMs.append(grad_cam.detach().cpu())
    y_hat.append((F.sigmoid(logits) > 0.5).float())
    cnn.zero_grad() 

  y_hat = torch.cat(y_hat).float().detach()
  Vgrads = torch.cat(Vgrads).float().detach()
  GradCAMs = torch.cat(GradCAMs).float().detach()
  
  return y_hat, Vgrads, GradCAMs





def predict_pinet(pinet, test_loader, device, return_logits=False):

  y_hat, Pi = [], []
  pinet.eval()

  with torch.no_grad():

    for (x, _) in test_loader:
      logits, pi, _ = pinet(x.to(device))
      y_hat.append(logits if return_logits else (F.sigmoid(logits) > 0.5).float())
      Pi.append(pi)

    y_hat = torch.cat(y_hat).float().detach()#.cpu()
    Pi = torch.cat(Pi).float().detach()#.cpu()
  
  return y_hat, Pi







def predict_on_piz(model, pi_loader, device):

  y_hat = []
  model.eval()

  with torch.no_grad():

    for (pi, x) in pi_loader:
      input = (pi.abs()).float().to(device) * x.unsqueeze(1).to(device)
      logits, _, _ = model(input.to(device))
      y_hat.append((F.sigmoid(logits) > 0.5).float())

    y_hat = torch.cat(y_hat, dim=0).float().detach()#.cpu()

  return y_hat




















def compute_metrics(Pi, Pi_star_test, y_test, thresholds, device):
    
  # Reshape once instead of for each threshold
  Pi_flat = Pi.reshape(Pi.size(0), -1).to(device)
  Pi_star_flat = Pi_star_test.reshape(Pi_star_test.size(0), -1).to(device)
  
  # Precompute common values for TDR
  mask_positive = (y_test == 1)
  Pi_star_flat_pos = Pi_star_flat[mask_positive].to(device)
  Pi_flat_pos = Pi_flat[mask_positive].to(device)
  pi_star_sum_pos = Pi_star_flat_pos.sum(dim=1).to(device)
  
  # Precompute common values for TAR
  one_minus_pi_star_flat = 1 - Pi_star_flat
  one_minus_pi_star_sum = one_minus_pi_star_flat.sum(dim=1)
  
  tdrs = []; tars = []
  for t in thresholds:
    # TDR calculation
    pi_t_pos = (Pi_flat_pos > t).float()
    tsd = (pi_t_pos * Pi_star_flat_pos).sum(dim=1) / pi_star_sum_pos
    tdr = tsd.mean().item()
    tdrs.append(tdr)
    
    # TAR calculation
    pi_t = (Pi_flat > t).float()
    one_minus_pi_t = 1 - pi_t
    tsa = (one_minus_pi_t * one_minus_pi_star_flat).sum(dim=1) / one_minus_pi_star_sum
    tar = tsa.mean().item()
    tars.append(tar)
  
  return np.array(tdrs), np.array(tars)





def Accuracy(y_true, y_pred):
  return (y_true == y_pred).sum().div(y_true.size(0))


  


def eval_detection(pi_, Pi_star_test, y_hat, y_test, verbose=True):
  acc = Accuracy(y_test, y_hat)
  tdr = TDR(pi_, Pi_star_test, y_test)
  tar = TAR(pi_, Pi_star_test)
  score = tdr*tar
  if verbose:
    print(f"Test Accuracy: {acc*100:.2f}%, TDR: {tdr*100:.2f}%, TAR: {tar*100:.2f}%, Score: {score:.3f}")
  return {'tar': tar, 'tdr': tdr, 'acc': acc, 'score': score}





def TDR(pi, pi_star, y):
  mask = (y==1)
  pi = pi.reshape(pi.size(0), -1)[mask]
  pi_star = pi_star.reshape(pi_star.size(0), -1)[mask]
  TSD = (pi*pi_star).sum(dim=1) / pi_star.sum(dim=1)
  return TSD.mean().item()




def TAR(pi, pi_star):
  pi = pi.reshape(pi.size(0), -1)
  pi_star = pi_star.reshape(pi_star.size(0), -1)
  TSA = ((1-pi)*(1-pi_star)).sum(dim=1) / (1-pi_star).sum(dim=1)
  return TSA.mean().item()





def store_maps_entry(
    model_kind: str,
    folder: str,
    name: str,
    MAPS: dict,
    Groups: dict,
    thresholds: np.ndarray,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    pick_best: bool = True,
    batch_size: int = 32,
):
    """
    Populate MAPS[name] with 'naive' and 'optimal' visualizations for:
      - model_kind='cnn'              (single CNN with GradCAM)
      - model_kind='pinet'            (single PiNet)
      - model_kind='pinet_ensemble'   (PiNet ensemble)
    """
    MAPS[name] = {}

    if model_kind == 'cnn':
        saved_models = glob(os.path.join(folder, '*.pth')); saved_models.sort()
        cfg = json.load(open(os.path.join(folder, 'cfg.json'), 'r'))

        # pick naive
        naive_scores = [res['naive']['score'] for res in Groups[name]]
        picked_naive = np.argmax(naive_scores) if pick_best else np.argsort(naive_scores)[14]

        cnn = CNN(**cfg).to(device)
        cnn.load_state_dict(torch.load(saved_models[picked_naive], map_location=device), strict=True)

        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        y_hat, _, cam = predict_cnn(cnn, test_loader, device=device)
        cam = cam / cam.max()
        camz = cam * X_test.unsqueeze(1)
        MAPS[name]['naive'] = {
            'pi': cam.detach().cpu().squeeze().numpy(),
            'piz': camz.detach().cpu().squeeze().numpy(),
            'score': naive_scores[picked_naive],
            'y_hat': y_hat.detach().cpu().numpy(),
        }

        # pick optimal
        optimal_scores = [res['optimal']['score'] for res in Groups[name]]
        picked_optimal = np.argmax(optimal_scores) if pick_best else np.argsort(optimal_scores)[14]
        t_opt = Groups[name][picked_optimal]['scores'].argmax()

        cnn.load_state_dict(torch.load(saved_models[picked_optimal], map_location=device), strict=True)
        y_hat, _, cam = predict_cnn(cnn, test_loader, device=device)
        cam = cam / cam.max()
        cam = (cam > thresholds[t_opt]).float()
        camz = cam * X_test.unsqueeze(1)
        MAPS[name]['optimal'] = {
            'pi': cam.detach().cpu().squeeze().numpy(),
            'piz': camz.detach().cpu().squeeze().numpy(),
            'score': optimal_scores[picked_optimal],
            'y_hat': y_hat.detach().cpu().numpy(),
        }

    elif model_kind == 'pinet':
        saved_models = glob(os.path.join(folder, '*.pth')); saved_models.sort()
        cfg = json.load(open(os.path.join(folder, 'cfg.json'), 'r'))
        cfg['activation'] = nn.Sigmoid() if cfg['activation'] == 'sigmoid' else nn.Tanh() if cfg['activation'] == 'tanh' else nn.Identity()

        # pick naive
        naive_scores = [res['naive']['score'] for res in Groups[name]]
        picked_naive = np.argmax(naive_scores) if pick_best else np.argsort(naive_scores)[14]

        pinet = BinaryConvPiNet(**cfg).to(device)
        pinet.load_state_dict(torch.load(saved_models[picked_naive], map_location=device), strict=True)
        logits, pi, _ = pinet(X_test.to(device))
        y_hat = (logits > 0.5).float()
        pi = pi / pi.max()
        piz = pi * X_test.unsqueeze(1)
        MAPS[name]['naive'] = {
            'pi': pi.detach().cpu().squeeze().numpy(),
            'piz': piz.detach().cpu().squeeze().numpy(),
            'score': naive_scores[picked_naive],
            'y_hat': y_hat.detach().cpu().numpy(),
        }

        # pick optimal
        optimal_scores = [res['optimal']['score'] for res in Groups[name]]
        picked_optimal = np.argmax(optimal_scores) if pick_best else np.argsort(optimal_scores)[14]
        t_opt = Groups[name][picked_optimal]['scores'].argmax()

        pinet.load_state_dict(torch.load(saved_models[picked_optimal], map_location=device), strict=True)
        logits, pi, _ = pinet(X_test.to(device))
        y_hat = (logits > 0.5).float()
        pi = pi / pi.max()
        pi = (pi > thresholds[t_opt]).float()
        piz = pi * X_test.unsqueeze(1)
        MAPS[name]['optimal'] = {
            'pi': pi.detach().cpu().squeeze().numpy(),
            'piz': piz.detach().cpu().squeeze().numpy(),
            'score': optimal_scores[picked_optimal],
            'y_hat': y_hat.detach().cpu().numpy(),
        }

    elif model_kind == 'pinet_ensemble':
        sub_folders = glob(os.path.join(folder, 'ensemble*')); sub_folders.sort()
        cfg = json.load(open(os.path.join(folder, 'cfg.json'), 'r'))
        cfg['activation'] = nn.Sigmoid() if cfg['activation'] == 'sigmoid' else nn.Tanh() if cfg['activation'] == 'tanh' else nn.Identity()

        # pick naive
        naive_scores = [res['naive']['score'] for res in Groups[name]]
        picked_naive = np.argmax(naive_scores) if pick_best else np.argsort(naive_scores)[14]
        pinet = BinaryConvPiNet(**cfg).to(device)
        indiv_models = glob(os.path.join(sub_folders[picked_naive], '*.pth')); indiv_models.sort()

        logits = torch.zeros_like(y_test).to(device)
        pi = torch.zeros_like(X_test.unsqueeze(1)).to(device)
        for mpath in indiv_models:
            pinet.load_state_dict(torch.load(mpath, map_location=device), strict=True)
            l, p, _ = pinet(X_test.to(device))
            logits += l / len(indiv_models)
            pi += pinet.b**2 * p / len(indiv_models)
        y_hat = (logits > 0.5).float()
        pi = pi / pi.max()
        piz = pi * X_test.unsqueeze(1)
        MAPS[name]['naive'] = {
            'pi': pi.detach().cpu().squeeze().numpy(),
            'piz': piz.detach().cpu().squeeze().numpy(),
            'score': naive_scores[picked_naive],
            'y_hat': y_hat.detach().cpu().numpy(),
        }

        # pick optimal
        optimal_scores = [res['optimal']['score'] for res in Groups[name]]
        picked_optimal = np.argmax(optimal_scores) if pick_best else np.argsort(optimal_scores)[14]
        indiv_models = glob(os.path.join(sub_folders[picked_optimal], '*.pth')); indiv_models.sort()

        logits = torch.zeros_like(y_test).to(device)
        pi = torch.zeros_like(X_test.unsqueeze(1)).to(device)
        for mpath in indiv_models:
            pinet.load_state_dict(torch.load(mpath, map_location=device), strict=True)
            l, p, _ = pinet(X_test.to(device))
            logits += l / len(indiv_models)
            pi += pinet.b**2 * p / len(indiv_models)
        y_hat = (logits > 0.5).float()
        pi = pi / pi.max()
        piz = pi * X_test.unsqueeze(1)
        MAPS[name]['optimal'] = {
            'pi': pi.detach().cpu().squeeze().numpy(),
            'piz': piz.detach().cpu().squeeze().numpy(),
            'score': optimal_scores[picked_optimal],
            'y_hat': y_hat.detach().cpu().numpy(),
        }

    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")