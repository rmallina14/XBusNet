import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def compute_segmentation_scores(pred: Tensor, label: Tensor, num_classes: int = 2):
    """
    Updated name of calculate_metrics.
    Returns only the scores actually used in train/validation.
    """

    intersect_area, pred_area, label_area, _ = calculate_area(pred, label, num_classes)

    _, acc = accuracy(intersect_area, pred_area)
    _, meaniou = mean_iou(intersect_area, pred_area, label_area)
    _, meandice = mean_dice(intersect_area, pred_area, label_area)
    _, meanf1 = f1_score(intersect_area, pred_area, label_area)
    meanfpr, meanfnr = fpr_fnr(intersect_area, pred_area, label_area)

    # return only these (all others removed)
    return acc, meaniou, meandice, meanf1, meanfpr, meanfnr


def calculate_area(pred: Tensor, label: Tensor, num_classes: int = 2, threshold: float = 0.5):
    pred = pred.detach().cpu()
    label = label.detach().cpu()

    if label.ndim == 3:  # [N, H, W]
        label = F.one_hot(torch.clamp(label, max=num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    elif label.ndim == 4 and label.shape[1] == 1:  # [N, 1, H, W]
        label = label.squeeze(1)
        label = F.one_hot(torch.clamp(label, max=num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    else:
        raise ValueError(f"Unexpected label shape: {label.shape}")

    pred_binary = (pred > threshold).float()
    pred_binary = torch.cat((1 - pred_binary, pred_binary), dim=1)  # N x 2 x H x W

    pred_area, label_area, intersect_area = [], [], []
    pred_save = None

    for i in range(num_classes):
        pred_i = pred_binary[:, i, :, :]
        label_i = label[:, i, :, :]
        if i == 1:
            pred_save = pred_i[0].numpy()
        pred_area.append(torch.sum(pred_i).unsqueeze(0))
        label_area.append(torch.sum(label_i).unsqueeze(0))
        intersect_area.append(torch.sum(pred_i * label_i).unsqueeze(0))

    pred_area = torch.cat(pred_area)
    label_area = torch.cat(label_area)
    intersect_area = torch.cat(intersect_area)

    return intersect_area, pred_area, label_area, pred_save


def mean_dice(intersect_area, pred_area, label_area):
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()
    union = pred_area + label_area
    class_dice = [(2 * inter / union[i]) if union[i] != 0 else 0 for i, inter in enumerate(intersect_area)]
    return np.array(class_dice), np.mean(class_dice)


def f1_score(intersect_area, pred_area, label_area):
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()
    precision = [(inter / pred_area[i]) if pred_area[i] != 0 else 0 for i, inter in enumerate(intersect_area)]
    recall = [(inter / label_area[i]) if label_area[i] != 0 else 0 for i, inter in enumerate(intersect_area)]
    f1 = [(2 * p * r / (p + r)) if (p + r) != 0 else 0 for p, r in zip(precision, recall)]
    return np.array(f1), np.mean(f1)


def mean_iou(intersect_area, pred_area, label_area):
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()
    union = pred_area + label_area - intersect_area
    class_iou = [(inter / union[i]) if union[i] != 0 else 0 for i, inter in enumerate(intersect_area)]
    return np.array(class_iou), np.mean(class_iou)


def accuracy(intersect_area, pred_area):
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    class_acc = [(inter / pred_area[i]) if pred_area[i] != 0 else 0 for i, inter in enumerate(intersect_area)]
    macc = np.sum(intersect_area) / np.sum(pred_area + 1e-10)
    return np.array(class_acc), macc


def fpr_fnr(intersect_area, pred_area, label_area):
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()

    fn = label_area - intersect_area
    tn = np.maximum(
        0,
        np.sum(label_area) + np.sum(pred_area) - 2 * np.sum(intersect_area)
    )

    fp = pred_area - intersect_area
    tp = intersect_area

    fpr = [fp[i] / (fp[i] + tn + 1e-10) if (fp[i] + tn) != 0 else 0 for i in range(len(fp))]
    fnr = [fn[i] / (fn[i] + tp[i] + 1e-10) if (fn[i] + tp[i]) != 0 else 0 for i in range(len(fn))]

    return np.mean(fpr), np.mean(fnr)
