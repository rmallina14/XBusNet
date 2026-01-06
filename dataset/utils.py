import numpy as np
import torch


def generate_visual_blend(img, seg, mode, image_size=224):
    """
    Build a visual prompt image given an input image, a segmentation mask,
    and a rendering mode.

    Args:
        img:  (C,H,W) image as torch.Tensor or np.ndarray
        seg:  (H,W) mask as torch.Tensor or np.ndarray
        mode: string specifying how to combine image and mask
        image_size: target size for some crop/blur modes

    Returns:
        List of one or two arrays/tensors, depending on the mode.
    """

    # ---------- helpers ----------

    def to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    def call_img_preprocess(blur=0, bg_fac=0.5, center_context=None, size=None):
        """
        Wrapper around evaluation_utils.build_visual_prompt_batch.

        Expects (meta, [img], [seg]) and returns a (1,C,H,W) tensor.
        """
        from evaluation_utils import build_visual_prompt_batch
        kwargs = dict(blur=blur, bg_fac=bg_fac)
        if center_context is not None:
            kwargs["center_context"] = center_context
        if size is not None:
            kwargs["image_size"] = size
        # build_visual_prompt_batch expects (meta, [img], [seg])
        return build_visual_prompt_batch((None, [img_t], [seg_t]), **kwargs)

    # modes that require tensors for evaluation_utils & math ops
    tensor_modes = {
        'blur_highlight', 'blur3_highlight', 'blur3_highlight01',
        'blur_highlight_random', 'crop', 'crop_blur_highlight',
        'crop_blur_highlight352'
    }

    if mode in tensor_modes:
        img_t = to_tensor(img)
        seg_t = to_tensor(seg)
    else:
        img_t, seg_t = img, seg

    # ---------- mode handling ----------

    if mode == 'overlay':
        img_t = to_tensor(img_t).float()
        seg_t = to_tensor(seg_t).float()
        out = img_t * seg_t.unsqueeze(0)
        return [out.float()]

    elif mode == 'highlight':
        img_t = to_tensor(img_t).float()
        seg_t = to_tensor(seg_t).float()
        mask_3c = seg_t.unsqueeze(0)
        out = img_t * mask_3c * 0.85 + 0.15 * img_t
        return [out.float()]

    elif mode == 'highlight2':
        img_t = to_tensor(img_t).float()
        seg_t = to_tensor(seg_t).float()
        base = img_t / 2.0
        out = (base + 0.1) * seg_t.unsqueeze(0) + 0.3 * base
        return [out.float()]

    elif mode == 'blur_highlight':
        proc = call_img_preprocess(blur=1, bg_fac=0.5)
        proc_np = proc[0].detach().cpu().numpy()
        return [proc_np - 0.01]

    elif mode == 'blur3_highlight':
        proc = call_img_preprocess(blur=3, bg_fac=0.5)
        proc_np = proc[0].detach().cpu().numpy()
        return [proc_np - 0.01]

    elif mode == 'blur3_highlight01':
        proc = call_img_preprocess(blur=3, bg_fac=0.1)
        proc_np = proc[0].detach().cpu().numpy()
        return [proc_np - 0.01]

    elif mode == 'blur_highlight_random':
        blur_val = int(torch.randint(low=0, high=3, size=(1,)).item())
        bg_fac = 0.1 + 0.8 * torch.rand(1).item()
        proc = call_img_preprocess(blur=blur_val, bg_fac=bg_fac)
        proc_np = proc[0].detach().cpu().numpy()
        return [proc_np - 0.01]

    elif mode == 'crop':
        proc = call_img_preprocess(blur=1, center_context=0.1, size=image_size)
        return [proc[0].detach().cpu().numpy()]

    elif mode == 'crop_blur_highlight':
        proc = call_img_preprocess(blur=3, center_context=0.1, bg_fac=0.1, size=image_size)
        return [proc[0].detach().cpu().numpy()]

    elif mode == 'crop_blur_highlight352':
        proc = call_img_preprocess(blur=3, center_context=0.1, bg_fac=0.1, size=352)
        return [proc[0].detach().cpu().numpy()]

    elif mode == 'shape':
        seg_np = to_numpy(seg_t)
        out = np.stack([seg_np] * 3).astype('float32')
        return [out]

    elif mode == 'concat':
        img_np = to_numpy(img_t)
        seg_np = to_numpy(seg_t)
        out = np.concatenate([img_np, seg_np[None, :, :]]).astype('float32')
        return [out]

    elif mode == 'image_only':
        img_np = to_numpy(img_t).astype('float32')
        return [img_np]

    elif mode == 'image_black':
        img_np = to_numpy(img_t).astype('float32')
        return [img_np * 0.0]

    elif mode is None:
        img_np = to_numpy(img_t).astype('float32')
        return [img_np]

    elif mode == 'separate':
        img_np = to_numpy(img_t).astype('float32')
        seg_np = to_numpy(seg_t).astype('int64')
        return [img_np, seg_np]

    elif mode == 'separate_img_black':
        img_np = to_numpy(img_t).astype('float32')
        seg_np = to_numpy(seg_t).astype('int64')
        return [img_np * 0.0, seg_np]

    elif mode == 'separate_seg_ones':
        img_np = to_numpy(img_t).astype('float32')
        seg_np = np.ones_like(to_numpy(seg_t)).astype('int64')
        return [img_np, seg_np]

    elif mode == 'separate_both_black':
        img_np = to_numpy(img_t).astype('float32')
        seg_np = to_numpy(seg_t).astype('int64')
        return [img_np * 0.0, seg_np * 0]

    else:
        raise ValueError(f"Invalid blend mode: {mode}")
