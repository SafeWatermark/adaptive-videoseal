"""
Whitebox attacks evaluator.

Evaluates watermarking models under whitebox attacks (WEvade) using both
standard and adaptive versions. Supports both removal and forgery attacks
with all aggregation strategies from the original VideoMarkBench.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Iterator, Tuple
import pandas as pd
import logging
from tqdm import tqdm
from skimage.metrics import structural_similarity
from scipy.stats import binom
from scipy.optimize import minimize

# Add submodule to path
submodule_path = Path(__file__).parent.parent / "videomarkbench"
sys.path.insert(0, str(submodule_path))

from aggregation import (  # noqa: E402
    logit_mean as _logit_mean,
    logit_g_median as _logit_g_median,
    bit_majority as _bit_majority,
    bit_g_median as _bit_g_median,
    bitacc_mean as _bitacc_mean,
    bitacc_median as _bitacc_median,
    detection_threshold as _detection_threshold,
    detection_majority as _detection_majority,
)


def project(param_data, backup, epsilon):
    """
    Project perturbation back to epsilon ball if it exceeds the upper bound.
    
    Args:
        param_data: Current perturbed data
        backup: Original data
        epsilon: Scaling factor (c in the code)
        
    Returns:
        Projected data
    """
    r = param_data - backup
    r = epsilon * r
    return backup + r


def _wevade_standard_removal_frame(
    watermarked_frame: torch.Tensor,
    gt_watermark: torch.Tensor,
    model,
    criterion: nn.Module,
    args: Dict,
) -> Tuple[torch.Tensor, Dict]:
    """
    Apply standard WEvade removal attack to a single frame.
    
    This is based on Wevade.py implementation with full PGD-style attack:
    - MSE Loss
    - Configurable iterations
    - L∞ norm projection
    - Early stopping
    
    Args:
        watermarked_frame: Single frame tensor (1, C, H, W)
        gt_watermark: Ground truth watermark (1, K)
        model: Watermarking model with detect() method
        criterion: Loss function (MSE Loss)
        args: Attack parameters:
            - iteration: Max iterations (default: 200)
            - epsilon: Early stopping threshold (default: 4/96)
            - alpha: Learning rate (default: 0.01)
            - rb: Perturbation bound (default: 0.1)
            - WEvade_type: 'WEvade-W-I' or 'WEvade-W-II' (default: 'WEvade-W-I')
            - secret_size: Watermark size (default: 96)
    
    Returns:
        Tuple of (attacked_frame, metrics_dict)
        - attacked_frame: Attacked frame tensor
        - metrics_dict: Dictionary with 'decoded_logits', 'success', etc.
    """
    watermarked_frame_cloned = watermarked_frame.clone()
    groundtruth_watermark = gt_watermark.clone()
    r = args.get('rb', 0.1)
    lr = args.get('alpha', 0.01)
    epsilon = args.get('epsilon', 4/96)
    iteration = args.get('iteration', 200)
    wevade_type = args.get('WEvade_type', 'WEvade-W-I')
    secret_size = args.get('secret_size', 96)
    success = False
    
    # WEvade_W_II target watermark selection
    if wevade_type == 'WEvade-W-II':
        target_watermark = torch.randint(0, 2, (1, secret_size), device=watermarked_frame.device).float() - 0.5
    # WEvade_W_I target watermark selection
    elif wevade_type == 'WEvade-W-I':
        # Call parent class detect() to bypass @torch.no_grad() decorator
        with torch.enable_grad():
            # Use super().detect() to bypass @torch.no_grad() in Videoseal.detect()
            # For single frame, we can call Wam.detect() directly
            if hasattr(model, 'detector'):
                # Direct detector call for gradient computation
                img_size = getattr(model, 'img_size', 256)
                imgs_res = watermarked_frame.clone()
                if watermarked_frame.shape[-2:] != (img_size, img_size):
                    imgs_res = F.interpolate(watermarked_frame, size=(img_size, img_size),
                                           mode='bilinear', align_corners=False, antialias=True)
                imgs_res = imgs_res.to(model.device)
                preds = model.detector(imgs_res).to(watermarked_frame.device)
                r_target_watermark = preds[:, 1:]  # Exclude detection bit
            else:
                # Fallback to detect() method
                r_target_watermark = model.detect(watermarked_frame, is_video=False)["preds"][:, 1:]
        r_target_watermark = (r_target_watermark > 0).float()
        target_watermark = 0.5 - groundtruth_watermark
    
    watermarked_frame_work = watermarked_frame.clone()
    
    for i in range(iteration):
        _watermarked_frame = watermarked_frame_work.requires_grad_(True)
        min_value, max_value = torch.min(_watermarked_frame), torch.max(_watermarked_frame)
        
        # Enable gradient computation for model
        # Call detector directly to bypass @torch.no_grad() decorator
        img_size = getattr(model, 'img_size', 256)
        imgs_res = _watermarked_frame.clone()
        if _watermarked_frame.shape[-2:] != (img_size, img_size):
            imgs_res = F.interpolate(_watermarked_frame, size=(img_size, img_size),
                                   mode='bilinear', align_corners=False, antialias=True)
        imgs_res = imgs_res.to(model.device)
        preds = model.detector(imgs_res).to(_watermarked_frame.device)
        decoded_watermark = preds[:, 1:]  # Exclude detection bit
        
        # Compute loss
        loss = criterion(decoded_watermark, target_watermark)
        grads = torch.autograd.grad(loss, _watermarked_frame, create_graph=False)
        
        with torch.no_grad():
            _watermarked_frame -= lr * torch.sign(grads[0])
            _watermarked_frame = torch.clamp(_watermarked_frame, min_value, max_value)
        
        del grads, loss
        
        # Projection
        perturbation_norm = torch.norm(_watermarked_frame - watermarked_frame_cloned, float('inf'))
        if perturbation_norm.cpu().detach().numpy() >= r:
            c = r / perturbation_norm
            _watermarked_frame = project(_watermarked_frame, watermarked_frame_cloned, c)
        
        # Check early stopping (no gradient needed)
        with torch.no_grad():
            imgs_res_check = _watermarked_frame.clone()
            if _watermarked_frame.shape[-2:] != (img_size, img_size):
                imgs_res_check = F.interpolate(_watermarked_frame, size=(img_size, img_size),
                                             mode='bilinear', align_corners=False, antialias=True)
            imgs_res_check = imgs_res_check.to(model.device)
            preds_check = model.detector(imgs_res_check).to(_watermarked_frame.device)
            decoded_watermark = preds_check[:, 1:]
        rounded_decoded_watermark = (decoded_watermark > 0).float()
        bit_acc_target = rounded_decoded_watermark.eq(target_watermark).sum().item() / secret_size
        watermarked_frame_work = _watermarked_frame
        
        # Early stopping
        if bit_acc_target >= 1 - epsilon:
            success = True
            break
        
        del _watermarked_frame, decoded_watermark, bit_acc_target
    
    post_processed_frame = watermarked_frame_work
    # Final detection (no gradient needed)
    with torch.no_grad():
        imgs_res_final = post_processed_frame.clone()
        if post_processed_frame.shape[-2:] != (img_size, img_size):
            imgs_res_final = F.interpolate(post_processed_frame, size=(img_size, img_size),
                                          mode='bilinear', align_corners=False, antialias=True)
        imgs_res_final = imgs_res_final.to(model.device)
        preds_final = model.detector(imgs_res_final).to(post_processed_frame.device)
        decoded_logits = preds_final[:, 1:].squeeze().detach().cpu()
    
    metrics = {
        'decoded_logits': decoded_logits,
        'success': success,
    }
    
    return post_processed_frame, metrics


def _wevade_standard_forgery_frame(
    original_frame: torch.Tensor,
    gt_watermark: torch.Tensor,
    model,
    criterion: nn.Module,
    args: Dict,
) -> Tuple[torch.Tensor, Dict]:
    """
    Apply standard WEvade forgery attack to a single frame.
    
    Args:
        original_frame: Single frame tensor (1, C, H, W)
        gt_watermark: Ground truth watermark (1, K)
        model: Watermarking model with detect() method
        criterion: Loss function (MSE Loss)
        args: Attack parameters (same as removal)
    
    Returns:
        Tuple of (attacked_frame, metrics_dict)
    """
    groundtruth_watermark = gt_watermark.clone()
    groundtruth_watermark -= 0.5
    original_frame_cloned = original_frame.clone()
    r = args.get('rb', 0.1)
    lr = args.get('alpha', 0.01)
    epsilon = args.get('epsilon', 4/96)
    iteration = args.get('iteration', 200)
    secret_size = args.get('secret_size', 96)
    success = False
    
    original_frame_work = original_frame.clone()
    
    for i in range(iteration):
        _original_frame = original_frame_work.requires_grad_(True)
        min_value, max_value = torch.min(_original_frame), torch.max(_original_frame)
        
        # Enable gradient computation for model
        # Call detector directly to bypass @torch.no_grad() decorator
        img_size = getattr(model, 'img_size', 256)
        imgs_res = _original_frame.clone()
        if _original_frame.shape[-2:] != (img_size, img_size):
            imgs_res = F.interpolate(_original_frame, size=(img_size, img_size),
                                   mode='bilinear', align_corners=False, antialias=True)
        imgs_res = imgs_res.to(model.device)
        preds = model.detector(imgs_res).to(_original_frame.device)
        decoded_watermark = preds[:, 1:]  # Exclude detection bit
        
        # Compute loss
        loss = criterion(decoded_watermark, groundtruth_watermark)
        grads = torch.autograd.grad(loss, _original_frame, create_graph=False)
        
        with torch.no_grad():
            _original_frame -= lr * torch.sign(grads[0])
            _original_frame = torch.clamp(_original_frame, min_value, max_value)
        
        del grads, loss
        
        # Projection
        perturbation_norm = torch.norm(_original_frame - original_frame_cloned, float('inf'))
        if perturbation_norm.cpu().detach().numpy() >= r:
            c = r / perturbation_norm
            _original_frame = project(_original_frame, original_frame_cloned, c)
        
        # Check early stopping (no gradient needed)
        with torch.no_grad():
            imgs_res_check = _original_frame.clone()
            if _original_frame.shape[-2:] != (img_size, img_size):
                imgs_res_check = F.interpolate(_original_frame, size=(img_size, img_size),
                                             mode='bilinear', align_corners=False, antialias=True)
            imgs_res_check = imgs_res_check.to(model.device)
            preds_check = model.detector(imgs_res_check).to(_original_frame.device)
            decoded_watermark = preds_check[:, 1:]
        rounded_decoded_watermark = (decoded_watermark > 0).float()
        bit_acc_target = rounded_decoded_watermark.eq(groundtruth_watermark > 0).sum().item() / secret_size
        original_frame_work = _original_frame
        
        # Early stopping
        if bit_acc_target >= 1 - epsilon:
            success = True
            break
        
        del _original_frame, decoded_watermark, bit_acc_target
    
    post_processed_frame = original_frame_work
    # Final detection (no gradient needed)
    with torch.no_grad():
        img_size = getattr(model, 'img_size', 256)
        imgs_res_final = post_processed_frame.clone()
        if post_processed_frame.shape[-2:] != (img_size, img_size):
            imgs_res_final = F.interpolate(post_processed_frame, size=(img_size, img_size),
                                          mode='bilinear', align_corners=False, antialias=True)
        imgs_res_final = imgs_res_final.to(model.device)
        preds_final = model.detector(imgs_res_final).to(post_processed_frame.device)
        decoded_logits = preds_final[:, 1:].squeeze().detach().cpu()
    
    metrics = {
        'decoded_logits': decoded_logits,
        'success': success,
    }
    
    return post_processed_frame, metrics


def _wevade_adaptive_removal_frame(
    watermarked_frame: torch.Tensor,
    gt_watermark: torch.Tensor,
    model,
    args: Dict,
) -> Tuple[torch.Tensor, Dict]:
    """
    Apply adaptive WEvade removal attack to a single frame.
    
    This is based on Wevade_adaptive.py implementation:
    - Direct loss calculation (no MSE)
    - Fixed 20 iterations
    - Fixed learning rate 0.02
    - No L∞ projection
    - No early stopping
    
    Args:
        watermarked_frame: Single frame tensor (1, C, H, W)
        gt_watermark: Ground truth watermark (1, K)
        model: Watermarking model with detect() method
        args: Attack parameters:
            - WEvade_type: 'WEvade-W-I' or 'WEvade-W-II' (default: 'WEvade-W-I')
            - secret_size: Watermark size (default: 96)
    
    Returns:
        Tuple of (attacked_frame, metrics_dict)
    """
    watermarked_frame_cloned = watermarked_frame.clone()
    groundtruth_watermark = gt_watermark.clone()
    wevade_type = args.get('WEvade_type', 'WEvade-W-I')
    secret_size = args.get('secret_size', 96)
    
    # WEvade_W_II target watermark selection
    if wevade_type == 'WEvade-W-II':
        target_watermark = torch.randint(0, 2, (1, secret_size), device=watermarked_frame.device).float() - 0.5
    # WEvade_W_I target watermark selection
    elif wevade_type == 'WEvade-W-I':
        # For WEvade-W-I, we use groundtruth_watermark to target the actual embedded watermark
        # Note: r_target_watermark is computed but not used, matching standard WEvade behavior
        # Call detector directly to bypass @torch.no_grad() decorator (for consistency, though not used)
        img_size = getattr(model, 'img_size', 256)
        with torch.no_grad():
            imgs_res = watermarked_frame.clone()
            if watermarked_frame.shape[-2:] != (img_size, img_size):
                imgs_res = F.interpolate(watermarked_frame, size=(img_size, img_size),
                                       mode='bilinear', align_corners=False, antialias=True)
            imgs_res = imgs_res.to(model.device)
            preds = model.detector(imgs_res).to(watermarked_frame.device)
            r_target_watermark = preds[:, 1:]  # Exclude detection bit
        r_target_watermark = (r_target_watermark > 0).float()
        # Adaptive WEvade uses r_target_watermark (detected watermark) as per original implementation
        # This is different from standard WEvade which uses groundtruth_watermark
        target_watermark = 0.5 - r_target_watermark
    
    watermarked_frame_work = watermarked_frame.clone()
    
    # Fixed 20 iterations with fixed learning rate 0.02
    for _ in range(20):
        watermarked_frame_work = watermarked_frame_work.requires_grad_(True)
        
        # Enable gradient computation for model
        # Call detector directly to bypass @torch.no_grad() decorator
        img_size = getattr(model, 'img_size', 256)
        imgs_res = watermarked_frame_work.clone()
        if watermarked_frame_work.shape[-2:] != (img_size, img_size):
            imgs_res = F.interpolate(watermarked_frame_work, size=(img_size, img_size),
                                   mode='bilinear', align_corners=False, antialias=True)
        imgs_res = imgs_res.to(model.device)
        preds = model.detector(imgs_res).to(watermarked_frame_work.device)
        decoded_logits = preds[:, 1:]  # Exclude detection bit
        
        # Direct loss calculation
        loss = - (target_watermark * decoded_logits).sum()
        grads = torch.autograd.grad(loss, watermarked_frame_work, create_graph=False)
        
        with torch.no_grad():
            watermarked_frame_work = watermarked_frame_work - 0.02 * torch.sign(grads[0])
            watermarked_frame_work = torch.clamp(watermarked_frame_work, 0, 1)
    
    # Final detection (no gradient needed)
    with torch.no_grad():
        img_size = getattr(model, 'img_size', 256)
        imgs_res_final = watermarked_frame_work.clone()
        if watermarked_frame_work.shape[-2:] != (img_size, img_size):
            imgs_res_final = F.interpolate(watermarked_frame_work, size=(img_size, img_size),
                                          mode='bilinear', align_corners=False, antialias=True)
        imgs_res_final = imgs_res_final.to(model.device)
        preds_final = model.detector(imgs_res_final).to(watermarked_frame_work.device)
        decoded_logits = preds_final[:, 1:].squeeze().detach().cpu()
    
    metrics = {
        'decoded_logits': decoded_logits.numpy() if isinstance(decoded_logits, torch.Tensor) else decoded_logits,
    }
    
    return watermarked_frame_work, metrics


def _wevade_adaptive_forgery_frame(
    original_frame: torch.Tensor,
    gt_watermark: torch.Tensor,
    model,
    args: Dict,
) -> Tuple[torch.Tensor, Dict]:
    """
    Apply adaptive WEvade forgery attack to a single frame.
    
    Args:
        original_frame: Single frame tensor (1, C, H, W)
        gt_watermark: Ground truth watermark (1, K)
        model: Watermarking model with detect() method
        args: Attack parameters (same as adaptive removal)
    
    Returns:
        Tuple of (attacked_frame, metrics_dict)
    """
    groundtruth_watermark = gt_watermark.clone()
    target_watermark = groundtruth_watermark - 0.5
    original_frame_cloned = original_frame.clone()
    
    original_frame_work = original_frame.clone()
    
    # Fixed 20 iterations with fixed learning rate 0.02
    for _ in range(20):
        original_frame_work = original_frame_work.requires_grad_(True)
        
        # Enable gradient computation for model
        # Call detector directly to bypass @torch.no_grad() decorator
        img_size = getattr(model, 'img_size', 256)
        imgs_res = original_frame_work.clone()
        if original_frame_work.shape[-2:] != (img_size, img_size):
            imgs_res = F.interpolate(original_frame_work, size=(img_size, img_size),
                                   mode='bilinear', align_corners=False, antialias=True)
        imgs_res = imgs_res.to(model.device)
        preds = model.detector(imgs_res).to(original_frame_work.device)
        decoded_watermark = preds[:, 1:]  # Exclude detection bit
        
        # Direct loss calculation
        loss = - (target_watermark * decoded_watermark).sum()
        grads = torch.autograd.grad(loss, original_frame_work, create_graph=False)
        
        with torch.no_grad():
            original_frame_work = original_frame_work - 0.02 * torch.sign(grads[0])
            original_frame_work = torch.clamp(original_frame_work, 0, 1)
    
    # Final detection (no gradient needed)
    with torch.no_grad():
        img_size = getattr(model, 'img_size', 256)
        imgs_res_final = original_frame_work.clone()
        if original_frame_work.shape[-2:] != (img_size, img_size):
            imgs_res_final = F.interpolate(original_frame_work, size=(img_size, img_size),
                                          mode='bilinear', align_corners=False, antialias=True)
        imgs_res_final = imgs_res_final.to(model.device)
        preds_final = model.detector(imgs_res_final).to(original_frame_work.device)
        decoded_logits = preds_final[:, 1:].squeeze().detach().cpu()
    
    metrics = {
        'decoded_logits': decoded_logits.numpy() if isinstance(decoded_logits, torch.Tensor) else decoded_logits,
    }
    
    return original_frame_work, metrics


def _call_with_watermark_gt_tensor(
    func, decode: torch.Tensor, watermark_gt: torch.Tensor
) -> bool:
    """
    Call aggregation function with custom watermark_gt (GPU-optimized tensor version).
    
    This is adapted from common_evaluator.py to work with whitebox attack results.
    
    Args:
        func: Aggregation function
        decode: Logits tensor of shape (T, K)
        watermark_gt: Ground truth watermark tensor of shape (K,)
        
    Returns:
        Boolean indicating if watermark is detected
    """
    func_name = func.__name__
    
    # Convert logits to bits: positive -> 1, negative/zero -> 0
    # VideoSeal convention: logits > 0 -> bit 1, logits <= 0 -> bit 0
    predicted_bits = (decode > 0).long()  # 1 for positive, 0 for negative/zero
    
    # Ensure watermark_gt is on same device
    watermark_gt = watermark_gt.to(decode.device)
    
    if func_name == 'logit_mean':
        decoded = torch.mean(decode, dim=0)
        predicted = (decoded > 0).long()
        bitacc = torch.sum((watermark_gt == predicted).float()) / len(watermark_gt)
        return (bitacc >= 67/96).item()
    
    elif func_name == 'logit_g_median':
        # For geometric median, convert to numpy (scipy.optimize required)
        decode_np = decode.detach().cpu().numpy()
        watermark_gt_np = watermark_gt.detach().cpu().numpy()
        def objective(median):
            return np.sum(np.linalg.norm(decode_np - median, axis=1))
        initial_guess = np.mean(decode_np, axis=0)
        decoded = minimize(objective, initial_guess, method='Powell', tol=1e-5)
        predicted = (decoded.x > 0).astype(int)
        bitacc = np.sum((watermark_gt_np == predicted)) / len(watermark_gt_np)
        return bitacc >= 67/96
    
    elif func_name == 'bit_majority':
        decoded = (torch.sum(predicted_bits, dim=0) > (decode.shape[0] / 2)).long()
        bitacc = torch.sum((watermark_gt == decoded).float()) / len(watermark_gt)
        return (bitacc >= 67/96).item()
    
    elif func_name == 'bit_g_median':
        decoded = torch.median(predicted_bits.float(), dim=0)[0].long()
        bitacc = torch.sum((watermark_gt == decoded).float()) / len(watermark_gt)
        return (bitacc >= 67/96).item()
    
    elif func_name == 'bitacc_mean':
        watermark_gt_tiled = watermark_gt.unsqueeze(0).expand(decode.shape[0], -1)
        bitacc = torch.mean((watermark_gt_tiled == predicted_bits).float())
        return (bitacc >= 67/96).item()
    
    elif func_name == 'bitacc_median':
        watermark_gt_tiled = watermark_gt.unsqueeze(0).expand(decode.shape[0], -1)
        bitacc = torch.mean((watermark_gt_tiled == predicted_bits).float(), dim=1)
        median_bitacc = torch.median(bitacc)
        return (median_bitacc >= 67/96).item()
    
    elif func_name == 'detection_threshold':
        watermark_gt_tiled = watermark_gt.unsqueeze(0).expand(decode.shape[0], -1)
        bitacc = torch.mean((watermark_gt_tiled == predicted_bits).float(), dim=1)
        detected_frames = torch.sum((bitacc >= 67/96).long()).item()
        
        n = decode.shape[0]
        p = 0.0000661
        threshold = 10e-4
        k = n  # Default if no threshold found
        for i in range(n + 1):
            if binom.sf(i - 1, n, p) <= threshold:
                k = i
                break
        
        return detected_frames >= k
    
    elif func_name == 'detection_majority':
        watermark_gt_tiled = watermark_gt.unsqueeze(0).expand(decode.shape[0], -1)
        bitacc = torch.mean((watermark_gt_tiled == predicted_bits).float(), dim=1)
        detected_frames = torch.sum((bitacc >= 67/96).long()).item()
        return detected_frames > (decode.shape[0] // 2)
    
    else:
        # Fallback: convert to numpy and use original function
        decode_np = decode.detach().cpu().numpy()
        watermark_gt_np = watermark_gt.detach().cpu().numpy()
        return _call_with_watermark_gt_numpy(func, decode_np, watermark_gt_np)


def _call_with_watermark_gt_numpy(
    func, decode: np.ndarray, watermark_gt: np.ndarray
) -> bool:
    """
    Call aggregation function with custom watermark_gt (CPU/numpy version for fallback).
    """
    func_name = func.__name__
    predicted_bits = (decode > 0).astype(int)
    
    if func_name == 'logit_mean':
        decoded = np.mean(decode, axis=0)
        predicted = (decoded > 0).astype(int)
        bitacc = np.sum((watermark_gt == predicted)) / len(watermark_gt)
        return bitacc >= 67/96
    
    elif func_name == 'logit_g_median':
        def objective(median):
            return np.sum(np.linalg.norm(decode - median, axis=1))
        initial_guess = np.mean(decode, axis=0)
        decoded = minimize(objective, initial_guess, method='Powell', tol=1e-5)
        predicted = (decoded.x > 0).astype(int)
        bitacc = np.sum((watermark_gt == predicted)) / len(watermark_gt)
        return bitacc >= 67/96
    
    elif func_name == 'bit_majority':
        decoded = (np.sum(predicted_bits, axis=0) > (len(decode) / 2)).astype(int)
        bitacc = np.sum((watermark_gt == decoded)) / len(watermark_gt)
        return bitacc >= 67/96
    
    elif func_name == 'bit_g_median':
        decoded = np.median(predicted_bits, axis=0).astype(int)
        bitacc = np.sum((watermark_gt == decoded)) / len(watermark_gt)
        return bitacc >= 67/96
    
    elif func_name == 'bitacc_mean':
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits))
        return bitacc >= 67/96
    
    elif func_name == 'bitacc_median':
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits), axis=1)
        median_bitacc = np.median(bitacc)
        return median_bitacc >= 67/96
    
    elif func_name == 'detection_threshold':
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits), axis=1)
        detected_frames = np.sum(bitacc >= 67/96)
        
        n = decode.shape[0]
        p = 0.0000661
        threshold = 10e-4
        k = n
        for i in range(n + 1):
            if binom.sf(i - 1, n, p) <= threshold:
                k = i
                break
        
        return detected_frames >= k
    
    elif func_name == 'detection_majority':
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits), axis=1)
        detected_frames = np.sum(bitacc >= 67 / 96)
        return detected_frames > (decode.shape[0] // 2)
    
    else:
        # Fallback to original function
        return func(decode)


# Aggregation strategies
AGGREGATION_STRATEGIES = {
    "logit_mean": _logit_mean,
    "logit_g_median": _logit_g_median,
    "bit_majority": _bit_majority,
    "bit_g_median": _bit_g_median,
    "bitacc_mean": _bitacc_mean,
    "bitacc_median": _bitacc_median,
    "detection_threshold": _detection_threshold,
    "detection_majority": _detection_majority,
}


def evaluate_whitebox_attacks(
    model,
    dataset: Iterator[Tuple[torch.Tensor, Dict]],
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    wevade_version: str = "standard",
    wevade_iteration: int = 200,
    wevade_epsilon: float = 4/96,
    wevade_alpha: float = 0.01,
    wevade_rb: float = 0.1,
    wevade_type: str = "WEvade-W-I",
    secret_size: int = 96,
    evaluate_forgery: bool = True,
    kinetics_dataset: Optional[Iterator[Tuple[torch.Tensor, Dict]]] = None,
) -> pd.DataFrame:
    """
    Evaluate whitebox attacks (WEvade) on watermarking model.
    
    Supports both standard and adaptive WEvade implementations:
    - Standard: Full PGD-style attack with MSE loss, configurable parameters
    - Adaptive: Simplified attack with fixed parameters (20 iterations, lr=0.02)
    
    Args:
        model: VideoMarkBenchModel wrapper instance
        dataset: Iterator of (video_tensor, metadata_dict) tuples
        output_dir: Directory to save results
        logger: Optional logger
        max_videos: Maximum number of videos to evaluate (None = all)
        max_frames_per_video: Maximum frames per video (None = all)
        wevade_version: 'standard' or 'adaptive' (default: 'standard')
        wevade_iteration: Max iterations for standard version (default: 200)
        wevade_epsilon: Early stopping threshold (default: 4/96)
        wevade_alpha: Learning rate (default: 0.01)
        wevade_rb: Perturbation bound (default: 0.1)
        wevade_type: 'WEvade-W-I' or 'WEvade-W-II' (default: 'WEvade-W-I')
        secret_size: Watermark size (default: 96)
        evaluate_forgery: Whether to evaluate forgery attacks (default: True)
        
    Returns:
        DataFrame with evaluation results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Prepare attack arguments
    attack_args = {
        'iteration': wevade_iteration,
        'epsilon': wevade_epsilon,
        'alpha': wevade_alpha,
        'rb': wevade_rb,
        'WEvade_type': wevade_type,
        'secret_size': secret_size,
    }
    
    # Select attack functions based on version
    if wevade_version == "standard":
        removal_fn = _wevade_standard_removal_frame
        forgery_fn = _wevade_standard_forgery_frame
        criterion = nn.MSELoss().to(model.device)
        logger.info(f"Using standard WEvade (iterations={wevade_iteration}, alpha={wevade_alpha}, rb={wevade_rb})")
    elif wevade_version == "adaptive":
        removal_fn = _wevade_adaptive_removal_frame
        forgery_fn = _wevade_adaptive_forgery_frame
        criterion = None  # Not used in adaptive version
        logger.info("Using adaptive WEvade (fixed 20 iterations, lr=0.02)")
    else:
        raise ValueError(f"Unknown wevade_version: {wevade_version}. Must be 'standard' or 'adaptive'")
    
    results = []
    device = model.device
    
    # Get actual watermark size from model
    try:
        # Try to get random message from model to determine actual size
        test_msg = model.model.get_random_msg()  # (1, K)
        actual_secret_size = test_msg.shape[1]
        if secret_size != actual_secret_size:
            logger.warning(
                f"Secret size mismatch: requested {secret_size}, but model uses {actual_secret_size}. "
                f"Using model's actual size: {actual_secret_size}"
            )
            secret_size = actual_secret_size
            # Update attack_args
            attack_args['secret_size'] = secret_size
    except Exception as e:
        logger.warning(f"Could not determine model's secret size, using provided {secret_size}: {e}")
    
    # Generate ground truth watermark pattern (alternating 0,1)
    gt_watermark = torch.zeros(1, secret_size, device=device)
    gt_watermark[:, 1::2] = 1  # Set every other bit to 1: [0,1,0,1,...]
    
    # Convert to long format for aggregation: (K,)
    embedded_message_long = gt_watermark.squeeze(0).long()  # Shape: (K,)
    
    video_count = 0
    for video, metadata in tqdm(dataset, desc="Whitebox evaluation"):
        if max_videos is not None and video_count >= max_videos:
            break
        
        video_name = metadata.get("video_name", f"video_{video_count}")
        video = video.to(device)
        
        # Limit frames if specified
        if max_frames_per_video is not None:
            video = video[:max_frames_per_video]
        
        T = video.shape[0]
        logger.debug(f"Processing {video_name} with {T} frames")
        
        # ===== REMOVAL ATTACK =====
        # Embed watermark first
        with torch.no_grad():
            embed_result = model.embed(video, message=gt_watermark)
            watermarked_video = embed_result["watermarked_video"]  # (T, C, H, W)
        
        # Attack each frame
        attacked_frames_logits = []
        for frame_idx in range(T):
            # Clone frame and enable gradient for attack
            frame = watermarked_video[frame_idx:frame_idx+1].clone().detach().requires_grad_(True)  # (1, C, H, W)
            
            try:
                if wevade_version == "standard":
                    attacked_frame, frame_metrics = removal_fn(
                        frame, gt_watermark, model.model, criterion, attack_args
                    )
                else:
                    attacked_frame, frame_metrics = removal_fn(
                        frame, gt_watermark, model.model, attack_args
                    )
                
                # Extract logits
                decoded_logits = frame_metrics['decoded_logits']
                if isinstance(decoded_logits, np.ndarray):
                    decoded_logits = torch.from_numpy(decoded_logits)
                attacked_frames_logits.append(decoded_logits)
                
            except Exception as e:
                logger.warning(f"Error in removal attack for {video_name} frame {frame_idx}: {e}")
                # Use original frame's logits as fallback
                with torch.no_grad():
                    logits = model.extract(frame)  # (1, K)
                    attacked_frames_logits.append(logits.squeeze(0).cpu())
        
        # Aggregate logits across frames
        if len(attacked_frames_logits) > 0:
            all_logits = torch.stack(attacked_frames_logits)  # (T, K)
            
            # Apply aggregation strategies
            for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                try:
                    detected = _call_with_watermark_gt_tensor(agg_func, all_logits, embedded_message_long)
                    results.append({
                        "video": video_name,
                        "attack_type": "removal",
                        "wevade_version": wevade_version,
                        "aggregation": agg_name,
                        "detected": bool(detected),
                        "type": "whitebox",
                    })
                except Exception as e:
                    logger.warning(f"Error in {agg_name} for {video_name} (removal): {e}")
        
        # ===== FORGERY ATTACK =====
        if evaluate_forgery:
            # Use kinetics dataset for forgery attack (as per original implementation)
            # If kinetics_dataset is provided, use it; otherwise use the same dataset
            if kinetics_dataset is not None:
                # Get next video from kinetics dataset
                try:
                    kinetics_video, kinetics_metadata = next(kinetics_dataset)
                    original_video = kinetics_video.to(device)
                    video_name_forgery = kinetics_metadata.get("video_name", f"kinetics_{video_count}")
                    T_forgery = original_video.shape[0]
                    if max_frames_per_video is not None:
                        T_forgery = min(T_forgery, max_frames_per_video)
                        original_video = original_video[:T_forgery]
                except StopIteration:
                    logger.warning(f"Kinetics dataset exhausted, skipping forgery for {video_name}")
                    continue
            else:
                # Fallback: use same dataset (for backward compatibility)
                original_video = video
                video_name_forgery = video_name
                T_forgery = T
            
            # Attack each frame
            attacked_frames_logits = []
            for frame_idx in range(T_forgery):
                # Clone frame and enable gradient for attack
                frame = original_video[frame_idx:frame_idx+1].clone().detach().requires_grad_(True)  # (1, C, H, W)
                
                try:
                    if wevade_version == "standard":
                        attacked_frame, frame_metrics = forgery_fn(
                            frame, gt_watermark, model.model, criterion, attack_args
                        )
                    else:
                        attacked_frame, frame_metrics = forgery_fn(
                            frame, gt_watermark, model.model, attack_args
                        )
                    
                    # Extract logits
                    decoded_logits = frame_metrics['decoded_logits']
                    if isinstance(decoded_logits, np.ndarray):
                        decoded_logits = torch.from_numpy(decoded_logits)
                    attacked_frames_logits.append(decoded_logits)
                    
                except Exception as e:
                    logger.warning(f"Error in forgery attack for {video_name_forgery} frame {frame_idx}: {e}")
                    # Use original frame's logits as fallback
                    with torch.no_grad():
                        logits = model.extract(frame)  # (1, K)
                        attacked_frames_logits.append(logits.squeeze(0).cpu())
            
            # Aggregate logits across frames
            if len(attacked_frames_logits) > 0:
                all_logits = torch.stack(attacked_frames_logits)  # (T, K)
                
                # Apply aggregation strategies
                for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                    try:
                        detected = _call_with_watermark_gt_tensor(agg_func, all_logits, embedded_message_long)
                        results.append({
                            "video": video_name_forgery,
                            "attack_type": "forgery",
                            "wevade_version": wevade_version,
                            "aggregation": agg_name,
                            "detected": bool(detected),
                            "type": "whitebox",
                        })
                    except Exception as e:
                        logger.warning(f"Error in {agg_name} for {video_name} (forgery): {e}")
        
        video_count += 1
    
    # Save results
    df = pd.DataFrame(results)
    if len(df) > 0:
        results_file = output_dir / "whitebox_results.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"Saved whitebox results to {results_file}")
        
        # Calculate and save statistics
        from evaluate.utils import calculate_statistics
        stats = calculate_statistics(df, logger)
        stats_file = output_dir / "whitebox_stats.json"
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved whitebox statistics to {stats_file}")
    else:
        logger.warning("No results to save")
    
    return df

