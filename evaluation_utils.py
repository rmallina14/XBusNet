import numpy as np
import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader

from general_utils import load_model


# ---------------------- Normalization helpers ----------------------


def denorm(img):
    """
    Undo ImageNet-style normalization on a tensor or numpy array.
    """
    np_input = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        np_input = True

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    img_denorm = (img * std[:, None, None]) + mean[:, None, None]

    if np_input:
        img_denorm = np.clip(img_denorm.numpy(), 0, 1)
    else:
        img_denorm = torch.clamp(img_denorm, 0, 1)

    return img_denorm


def norm(img):
    """
    Apply ImageNet-style normalization to a tensor.
    """
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    return (img - mean[:, None, None]) / std[:, None, None]


# ---------------------- Metric curves ----------------------


def fast_iou_curve(p, g):
    """
    Approximate IoU as a function of threshold over sorted predictions.
    """
    g = g[p.sort().indices]
    p = torch.sigmoid(p.sort().values)

    scores = []
    vals = np.linspace(0, 1, 50)

    for q in vals:
        n = int(len(g) * q)

        valid = torch.where(p > q)[0]
        if len(valid) > 0:
            n = int(valid[0])
        else:
            n = len(g)

        fn = g[:n].sum()
        tn = n - fn
        tp = g[n:].sum()
        fp = len(g) - n - tp

        iou = tp / (tp + fn + fp + 1e-10)

        # not used downstream, but kept for compatibility
        _precision = tp / (tp + fp + 1e-10)
        _recall = tp / (tp + fn + 1e-10)

        scores.append(float(iou))

    return vals, scores


def fast_rp_curve(p, g):
    """
    Approximate precision–recall curve from sorted predictions.
    """
    g = g[p.sort().indices]
    p = torch.sigmoid(p.sort().values)

    precisions, recalls = [], []

    # sample thresholds from sorted predictions
    for q in p[::100000]:
        valid = torch.where(p > q)[0]
        if len(valid) > 0:
            n = int(valid[0])
        else:
            n = len(g)

        fn = g[:n].sum()
        tn = n - fn
        tp = g[n:].sum()
        fp = len(g) - n - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        precisions.append(float(precision))
        recalls.append(float(recall))

    return recalls, precisions


# ---------------------- Image processing ----------------------


def img_preprocess(batch,
                   blur=0,
                   grayscale=False,
                   center_context=None,
                   rect=False,
                   rect_color=(255, 0, 0),
                   rect_width=2,
                   brightness=1.0,
                   bg_fac=1.0,
                   colorize=False,
                   outline=False,
                   image_size=224):
    """
    Build a visually emphasized image given (meta, images, masks) batch.

    batch: (meta, imgs, masks) where imgs, masks are lists of tensors/arrays:
           imgs[i] : (C,H,W), masks[i] : (H,W)
    """
    import cv2

    rw = rect_width
    out = []

    # batch structure: (meta, imgs, masks)
    imgs = batch[1]
    masks = batch[2]

    for img, mask in zip(imgs, masks):
        img = img.cpu() if isinstance(img, torch.Tensor) else torch.from_numpy(img)
        mask = mask.cpu() if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)

        img = img * brightness
        img_bl = img

        if blur > 0:
            # Gaussian blur in HWC then back to CHW
            img_bl = torch.from_numpy(
                cv2.GaussianBlur(img.permute(1, 2, 0).numpy(), (15, 15), blur)
            ).permute(2, 0, 1)

        if grayscale:
            img_bl = img_bl[1][None]

        # foreground: img * mask, background: blurred scaled by bg_fac
        img_inp = img * mask + bg_fac * img_bl * (1 - mask)

        if rect:
            _, bbox = crop_mask(img, mask, context=0.1)
            color = torch.tensor(rect_color, dtype=img_inp.dtype)[:, None, None]
            # left / right vertical bars
            img_inp[:, bbox[2]: bbox[3], max(0, bbox[0] - rw):bbox[0] + rw] = color
            img_inp[:, bbox[2]: bbox[3], max(0, bbox[1] - rw):bbox[1] + rw] = color
            # top / bottom horizontal bars
            img_inp[:, max(0, bbox[2] - 1): bbox[2] + rw, bbox[0]:bbox[1]] = color
            img_inp[:, max(0, bbox[3] - 1): bbox[3] + rw, bbox[0]:bbox[1]] = color

        if center_context is not None:
            img_inp = object_crop(img_inp, mask, context=center_context, image_size=image_size)

        if colorize:
            img_gray = denorm(img)
            img_gray = cv2.cvtColor(img_gray.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
            img_gray = torch.stack([torch.from_numpy(img_gray)] * 3)
            img_inp = (
                torch.tensor([1, 0.2, 0.2])[:, None, None] * img_gray * mask
                + bg_fac * img_gray * (1 - mask)
            )
            img_inp = norm(img_inp)

        if outline:
            cont = cv2.findContours(mask.byte().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline_img = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(outline_img, cont[0], -1, thickness=5, color=(255, 255, 255))
            outline_img = torch.stack([torch.from_numpy(outline_img)] * 3).float() / 255.0
            img_inp = (
                torch.tensor([1, 0, 0])[:, None, None] * outline_img
                + denorm(img_inp) * (1 - outline_img)
            )
            img_inp = norm(img_inp)

        out.append(img_inp)

    return torch.stack(out)


def object_crop(img, mask, context=0.0, square=False, image_size=224):
    """
    Crop image around mask with optional context percent, then pad to square and resize.
    """
    img_crop, bbox = crop_mask(img, mask, context=context, square=square)
    img_crop = pad_to_square(img_crop, channel_dim=0)
    img_crop = torch.nn.functional.interpolate(
        img_crop.unsqueeze(0), (image_size, image_size)
    ).squeeze(0)
    return img_crop


def crop_mask(img, mask, context=0.0, square=False):
    """
    Compute tight bounding box around mask and crop img accordingly.
    """
    assert img.shape[1:] == mask.shape

    bbox = [
        mask.max(0).values.argmax(),
        mask.size(0) - mask.max(0).values.flip(0).argmax(),
    ]
    bbox += [
        mask.max(1).values.argmax(),
        mask.size(1) - mask.max(1).values.flip(0).argmax(),
    ]
    bbox = [int(x) for x in bbox]

    width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])

    if square:
        bbox[0] = int(max(0, bbox[0] - context * height))
        bbox[1] = int(min(mask.size(0), bbox[1] + context * height))
        bbox[2] = int(max(0, bbox[2] - context * width))
        bbox[3] = int(min(mask.size(1), bbox[3] + context * width))

        width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])
        if height > width:
            bbox[2] = int(max(0, (bbox[2] - 0.5 * height)))
            bbox[3] = bbox[2] + height
        else:
            bbox[0] = int(max(0, (bbox[0] - 0.5 * width)))
            bbox[1] = bbox[0] + width
    else:
        bbox[0] = int(max(0, bbox[0] - context * height))
        bbox[1] = int(min(mask.size(0), bbox[1] + context * height))
        bbox[2] = int(max(0, bbox[2] - context * width))
        bbox[3] = int(min(mask.size(1), bbox[3] + context * width))

    img_crop = img[:, bbox[2]: bbox[3], bbox[0]: bbox[1]]
    return img_crop, bbox


def pad_to_square(img, channel_dim=2, fill=0):
    """
    Add padding such that a squared image is returned.
    """
    from torchvision.transforms.functional import pad

    if channel_dim == 2:
        img = img.permute(2, 0, 1)
    elif channel_dim == 0:
        pass
    else:
        raise ValueError('invalid channel_dim')

    h, w = img.shape[1:]
    pady1 = pady2 = padx1 = padx2 = 0

    if h > w:
        padx1 = (h - w) // 2
        padx2 = h - w - padx1
    elif w > h:
        pady1 = (w - h) // 2
        pady2 = w - h - pady1

    img_padded = pad(img, padding=(padx1, pady1, padx2, pady2), padding_mode='constant')

    if channel_dim == 2:
        img_padded = img_padded.permute(1, 2, 0)

    return img_padded


# ---------------------- Qualitative plots ----------------------


def split_sentence(inp, limit=9):
    t_new, current_len = [], 0
    words = inp.split(' ')
    for k, t in enumerate(words):
        current_len += len(t) + 1
        t_new.append(t + ' ')
        if current_len > limit and k != len(words) - 1:
            current_len = 0
            t_new.append('\n')

    return ''.join(t_new)


from matplotlib import pyplot as plt


def plot(imgs, *preds, labels=None, scale=1, cmap=plt.cm.magma, aps=None, gt_labels=None, vmax=None):
    row_off = 0 if labels is None else 1
    _, ax = plt.subplots(
        len(imgs) + row_off,
        1 + len(preds),
        figsize=(scale * float(1 + 2 * len(preds)), scale * float(len(imgs) * 2))
    )
    [a.axis('off') for a in ax.flatten()]

    if labels is not None:
        for j in range(len(labels)):
            t_new = split_sentence(labels[j], limit=6)
            ax[0, 1 + j].text(0.5, 0.1, t_new, ha='center', fontsize=3 + 10 * scale)

    for i in range(len(imgs)):
        ax[i + row_off, 0].imshow(imgs[i])
        for j in range(len(preds)):
            img = preds[j][i][0].detach().cpu().numpy()

            if gt_labels is not None and labels[j] == gt_labels[i]:
                edgecolor = 'red'
                if aps is not None:
                    ax[i + row_off, 1 + j].text(
                        30, 70, 'AP: {:.3f}'.format(aps[i]), color='red', fontsize=8
                    )
            else:
                edgecolor = 'k'

            rect = plt.Rectangle(
                [0, 0], img.shape[0], img.shape[1],
                facecolor="none", edgecolor=edgecolor, linewidth=3
            )
            ax[i + row_off, 1 + j].add_patch(rect)

            if vmax is None:
                this_vmax = 1
            elif vmax == 'per_prompt':
                this_vmax = max([preds[j][_i][0].max() for _i in range(len(imgs))])
            elif vmax == 'per_image':
                this_vmax = max([preds[_j][i][0].max() for _j in range(len(preds))])
            else:
                this_vmax = 1

            ax[i + row_off, 1 + j].imshow(img, vmin=0, vmax=this_vmax, cmap=cmap)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)


# ---------------------- Compatibility helper ----------------------


def build_visual_prompt_batch(batch,
                              blur=0,
                              grayscale=False,
                              center_context=None,
                              rect=False,
                              rect_color=(255, 0, 0),
                              rect_width=2,
                              brightness=1.0,
                              bg_fac=1.0,
                              colorize=False,
                              outline=False,
                              image_size=224):
    """
    Thin wrapper around img_preprocess for legacy code that imports
    `build_visual_prompt_batch` from this module.
    """
    return img_preprocess(
        batch,
        blur=blur,
        grayscale=grayscale,
        center_context=center_context,
        rect=rect,
        rect_color=rect_color,
        rect_width=rect_width,
        brightness=brightness,
        bg_fac=bg_fac,
        colorize=colorize,
        outline=outline,
        image_size=image_size,
    )
