"""Multi-task loss: cross-entropy on the per-sample phase labels + magnitude-weighted Huber losses on the
window-level intensity and magnitude (P windows only)."""
import torch
import torch.nn as nn


def masked_scalar_huber_loss_int(pred, target, event_present, delta=0.5, eps=1e-6):
    """Huber loss on the intensity target, up-weighting strong shaking. pred, target, event_present: (B,)."""
    huber = nn.functional.huber_loss(pred, target, delta=delta, reduction="none")
    w = torch.ones_like(target)
    w = torch.where(target > 4.0, 2.0, w)
    w = torch.where(target > 6.0, 3.0, w)
    w = torch.where(target > 7.0, 5.0, w)
    w = torch.where(target > 8.0, 10, w)
    return (huber * event_present * w).sum() / ((event_present * w).sum() + eps)


def masked_scalar_huber_loss_mag(pred, target, event_present, delta=0.5, eps=1e-6):
    """Huber loss on the magnitude, up-weighting large events. pred, target, event_present: (B,)."""
    huber = nn.functional.huber_loss(pred, target, delta=delta, reduction="none")
    w = torch.ones_like(target)
    w = torch.where(target > 4.0, 2.0, w)
    w = torch.where(target > 5.0, 3.0, w)
    w = torch.where(target > 6.0, 5.0, w)
    return (huber * event_present * w).sum() / ((event_present * w).sum() + eps)


def total_loss(out, y_mag, y_int, y_det, ce, lam, delta=0.5):
    """out = model(x); y_det: (B, L) class per sample; lam = (lambda_det, lambda_int, lambda_mag).
    Regression targets count only for windows with an event (y_mag > 0, i.e. P windows)."""
    mag_p, int_p, det_p = out
    present = (y_mag > 0).float()
    l_mag = masked_scalar_huber_loss_mag(mag_p, y_mag, present, delta=delta)
    l_int = masked_scalar_huber_loss_int(int_p, y_int, present, delta=delta)
    l_det = ce(det_p.permute(0, 2, 1), y_det)
    return lam[0] * l_det + lam[1] * l_int + lam[2] * l_mag, (l_det.item(), l_int.item(), l_mag.item())
