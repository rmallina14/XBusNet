import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from datasets.utils import generate_visual_blend


class XBusNetDataset(Dataset):
    """
    Dataset for XBusNet:
    
    - global_prompt: sentence-level descriptor (GFCP in the paper)
    - local_prompt:  lesion-specific phrase (LFP in the paper)
    - vis_image:     optional blended visual prompt (used only when with_visual=True)
    - vis_flag:      indicates whether visual prompt is valid

    Returns:
        (
            image, 
            global_prompt, 
            local_prompt, 
            vis_image, 
            vis_flag,
        ),
        (
            mask, 
            empty_tensor, 
            index_tensor
        )
    """

    def __init__(
        self,
        csv_file,
        image_dir,
        mask_dir,
        image_size=352,
        split="train",
        negative_prob=0.0,
        aug=False,
        with_visual=True,
        only_visual=False,
        mask="text_and_crop_blur_highlight352",
        **kwargs,
    ):
        self.image_size = image_size
        self.negative_prob = negative_prob
        self.aug = aug
        self.with_visual = with_visual
        self.only_visual = only_visual
        self.mask = mask
        self.split = split

        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.prompt_csv = csv_file

        df = pd.read_csv(self.prompt_csv)
        df = df.dropna(subset=["Image Name", "Sentence", "AGM_Prompt"])

        self.image_names = df["Image Name"].tolist()
        self.global_texts = df["Sentence"].tolist()    # global prompt (GFCP)
        self.local_texts = df["AGM_Prompt"].tolist()   # local prompt (LFP)

        # For sampling other images that share the same local prompt
        self.all_local_prompts = list(set(self.local_texts))
        self.samples_by_local_prompt = {p: [] for p in self.all_local_prompts}
        for i, p in enumerate(self.local_texts):
            self.samples_by_local_prompt[p].append(i)

        # Modes understood by generate_visual_blend
        self.visual_mode_priority = [
            "crop_blur_highlight352",
            "crop_blur_highlight",
            "crop",
            "blur3_highlight01",
            "blur3_highlight",
            "blur_highlight",
            "highlight",
        ]

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.image_names)

    def _resolve_visual_mode(self):
        """
        Map self.mask (e.g., 'text_and_crop_blur_highlight352')
        to one of the actual modes supported by generate_visual_blend.
        """
        m = self.mask
        for key in self.visual_mode_priority:
            if key in m:
                return key
        return "highlight"

    def load_image_mask(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".png", "_tumor.png"))

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # binarize if needed
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)

        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

        img = F.interpolate(
            img,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        mask = F.interpolate(
            mask,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )[0, 0]

        img = self.normalize(img)
        return img, mask

    def __getitem__(self, index):
        img, seg = self.load_image_mask(index)

        global_prompt = self.global_texts[index]
        local_prompt = self.local_texts[index]

        # Negative sampling: replace local prompt and zero-out mask
        active_local_prompt = local_prompt
        if self.negative_prob > 0 and torch.rand(1).item() < self.negative_prob:
            while True:
                new_lp = self.all_local_prompts[
                    torch.randint(len(self.all_local_prompts), (1,)).item()
                ]
                if new_lp != local_prompt:
                    active_local_prompt = new_lp
                    seg = torch.zeros_like(seg)
                    break

        # Visual prompt branch
        if self.with_visual:
            if (
                active_local_prompt in self.samples_by_local_prompt
                and len(self.samples_by_local_prompt[active_local_prompt]) > 1
            ):
                # pick another sample that shares the same local prompt
                idx = torch.randint(
                    len(self.samples_by_local_prompt[active_local_prompt]), (1,)
                ).item()
                other_index = self.samples_by_local_prompt[active_local_prompt][idx]

                img_s, seg_s = self.load_image_mask(other_index)

                mode = self._resolve_visual_mode()
                vis_image = generate_visual_blend(
                    img_s, seg_s, mode=mode, image_size=self.image_size
                )[0]

                if isinstance(vis_image, np.ndarray):
                    vis_image = torch.from_numpy(vis_image).float()

                vis_image_tensor = vis_image
                vis_flag = torch.tensor(True)
            else:
                vis_image_tensor = torch.zeros_like(img)
                vis_flag = torch.tensor(False)
        else:
            vis_image_tensor = torch.zeros_like(img)
            vis_flag = torch.tensor(False)

        seg = seg.unsqueeze(0).float()

        return (
            img,
            global_prompt,
            active_local_prompt,
            vis_image_tensor,
            vis_flag,
        ), (
            seg,
            torch.zeros(0),
            torch.tensor(index),
        )
