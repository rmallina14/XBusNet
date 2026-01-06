import os
import math
from os.path import isfile

import torch
from torch import nn
from torch.nn import functional as F
import clip
from torchvision.models import resnet50


# -------------------------------------------------------------------------
# Multi-head attention helper (used inside the CLIP visual transformer)
# -------------------------------------------------------------------------
def forward_multihead_attention(x, block, with_aff=False, attn_mask=None):
    """
    Lightweight multi-head self-attention step using a CLIP-style residual block.

    Args:
        x:        [L, B, C] sequence input.
        block:    CLIP transformer block with ln_1, attn, ln_2, mlp.
        with_aff: if True, also returns attention weights per head.
        attn_mask: optional tuple (mask_type, mask_tensor) where mask_tensor
                   is applied on the spatial tokens.
    """
    x_norm = block.ln_1(x)
    q, k, v = F.linear(
        x_norm, block.attn.in_proj_weight, block.attn.in_proj_bias
    ).chunk(3, dim=-1)

    tgt_len, bsz, embed_dim = q.size()
    num_heads = block.attn.num_heads
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5

    # reshape to (n_heads * B, L, head_dim)
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    q = q * scale
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # [n_heads*B, L, L]

    # optional masking
    if attn_mask is not None:
        mask_type, mask = attn_mask
        # mask has shape [B, L-1]; repeat per head
        n_heads_total = attn_output_weights.size(0) // mask.size(0)
        mask = mask.repeat(n_heads_total, 1)

        if mask_type == "cls_token":
            # affect only similarities w.r.t. CLS token
            attn_output_weights[:, 0, 1:] *= mask[None, ...]
        elif mask_type == "all":
            attn_output_weights[:, 1:, 1:] *= mask[:, None]

    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = block.attn.out_proj(attn_output)

    x = x + attn_output
    x = x + block.mlp(block.ln_2(x))

    if with_aff:
        return x, attn_output_weights
    return x


# -------------------------------------------------------------------------
# Global CLIP visual backbone (shared by Global Feature Extractor)
# -------------------------------------------------------------------------
class GlobalVisionBackbone(nn.Module):
    """
    Base CLIP backbone used inside the Global Feature Extractor.

    IMPORTANT:
    - Uses TorchScript CLIP (.pt) OFFLINE.
    - We only use:
        * encode_text  → for global prompts
        * encode_image → for visual global conditional
    - We DO NOT walk through transformer.resblocks (TorchScript limitation).
    """

    def __init__(
        self,
        version: str,
        reduce_cond: int,
        reduce_dim: int,
        prompt: str,
        n_tokens: int,
        clip_ckpt_path: str = "/home/mallinar/.cache/clip/ViT-B-16/ViT-B-16.pt",
    ):
        super().__init__()

        # ---- offline CLIP loading (TorchScript) ----
        if not os.path.isfile(clip_ckpt_path):
            raise FileNotFoundError(
                f"CLIP checkpoint not found at {clip_ckpt_path}. "
                f"Please copy ViT-B-16.pt there or update clip_ckpt_path."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = torch.jit.load(clip_ckpt_path, map_location=device).eval()

        # we no longer rely on internal .visual structure, but keep this for compatibility
        self.model = getattr(self.clip_model, "visual", None)

        self.version = version
        self.n_tokens = n_tokens  # kept for compatibility, not used directly

        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        # conditional projection (text → FiLM)
        self.reduce_cond_dim = reduce_cond
        if reduce_cond is not None:
            self.reduce_cond = nn.Linear(512, reduce_cond)
            for p in self.reduce_cond.parameters():
                p.requires_grad_(False)
            cond_dim = reduce_cond
        else:
            self.reduce_cond = None
            cond_dim = 512

        self.film_mul = nn.Linear(cond_dim, reduce_dim)
        self.film_add = nn.Linear(cond_dim, reduce_dim)

        # linear reduction kept for compatibility (not used in new visual path)
        self.reduce = nn.Linear(768, reduce_dim)

        self.prompt_list = ["{}"]  # simple template

        # precomputed prompts (optional)
        import pickle
        if isfile("precomputed_prompt_vectors.pickle"):
            precomp = pickle.load(open("precomputed_prompt_vectors.pickle", "rb"))
            self.precomputed_prompts = {k: torch.from_numpy(v) for k, v in precomp.items()}
        else:
            self.precomputed_prompts = dict()

        # optional shift for null-prompt variants
        self.shift_vector = None

    # ------------------------------ visual path -----------------------------
    def visual_forward(self, x_inp, *args, **kwargs):
        """
        Black-box visual forward using CLIP's encode_image.

        NOTE:
          - CLIP ViT-B/16 expects 224x224 inputs (14x14 patches).
          - Your ultrasound images are 352x352, so we must resize to 224x224
            or CLIP's positional embeddings will mismatch (485 vs 197 tokens).
        Returns:
            img_emb: [B, 512] CLIP image embedding
            activations: [] (kept for API compatibility)
            affinities:  [] (kept for API compatibility)
        """
        device = next(self.parameters()).device
        x = x_inp.to(device)

        # ensure 3-channel
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        with torch.no_grad():
            img_emb = self.clip_model.encode_image(x)  # [B,512]

        return img_emb, [], []


    # --------------------------- text / condition ---------------------------

    def sample_prompts(self, words, prompt_list=None):
        prompt_list = prompt_list if prompt_list is not None else self.prompt_list
        idx = torch.multinomial(torch.ones(len(prompt_list)), len(words), replacement=True)
        chosen = [prompt_list[i] for i in idx]
        return [p.format(w) for p, w in zip(chosen, words)]

    def compute_conditional(self, conditional):
        """
        Encode text prompts via CLIP to a [B,512] embedding.
        Supports:
          - single string
          - list/tuple of strings
          - cached precomputed prompts
        """
        dev = next(self.parameters()).device

        # list / tuple of strings
        if isinstance(conditional, (list, tuple)):
            text_tokens = clip.tokenize(conditional).to(dev)
            cond = self.clip_model.encode_text(text_tokens)

        else:
            # single string with possible precomputed cache
            if conditional in self.precomputed_prompts:
                cond = self.precomputed_prompts[conditional].float().to(dev)
            else:
                text_tokens = clip.tokenize([conditional]).to(dev)
                cond = self.clip_model.encode_text(text_tokens)[0]

        if self.shift_vector is not None:
            return cond + self.shift_vector
        return cond

    def get_cond_vec(self, conditional, batch_size):
        """
        Normalize different conditional formats into a [B, 512] tensor.
        """
        # single string
        if conditional is not None and isinstance(conditional, str):
            cond = self.compute_conditional(conditional)
            cond = cond.repeat(batch_size, 1)

        # list/tuple of strings
        elif conditional is not None and isinstance(conditional, (list, tuple)) and isinstance(
            conditional[0], str
        ):
            assert len(conditional) == batch_size
            cond = self.compute_conditional(conditional)

        # already a [B, D] tensor
        elif conditional is not None and isinstance(conditional, torch.Tensor) and conditional.ndim == 2:
            cond = conditional

        # image tensor → use visual branch to derive condition
        elif conditional is not None and isinstance(conditional, torch.Tensor):
            cond, _, _ = self.visual_forward(conditional)
        else:
            raise ValueError("invalid conditional")
        return cond


# -------------------------------------------------------------------------
# Global Feature Extractor (GFE) – CLIP-based branch
# -------------------------------------------------------------------------
class GlobalFeatureExtractor(GlobalVisionBackbone):
    """
    Global Feature Extractor (GFE).

    NEW DESIGN:
      - Uses CLIP encode_text for global prompt → cond [B,512]
      - Optionally uses CLIP encode_image for visual mix (train.py uses this)
      - Builds a spatial feature map from the input image via a small conv,
        modulated by FiLM from the text condition.
    """

    def __init__(
        self,
        version="ViT-B/16",
        extract_layers=(3, 7, 9),   # kept for config compatibility (not used)
        cond_layer=0,               # kept for compatibility
        reduce_dim=64,
        n_heads=4,                  # kept for config compatibility
        prompt="fixed",
        extra_blocks=0,             # not used in this simplified variant
        reduce_cond=None,
        fix_shift=False,            # kept for compatibility
        learn_trans_conv_only=False,
        limit_to_clip_only=False,
        upsample=False,
        add_calibration=False,
        rev_activations=False,
        trans_conv=None,
        n_tokens=None,
        complex_trans_conv=False,
        clip_ckpt_path="/home/mallinar/.cache/clip/ViT-B-16/ViT-B-16.pt",
    ):
        super().__init__(
            version=version,
            reduce_cond=reduce_cond,
            reduce_dim=reduce_dim,
            prompt=prompt,
            n_tokens=n_tokens,
            clip_ckpt_path=clip_ckpt_path,
        )

        self.extract_layers = tuple(extract_layers)
        self.cond_layer = cond_layer
        self.rev_activations = rev_activations

        # Simple image → feature projection (global branch backbone)
        self.image_proj = nn.Conv2d(3, reduce_dim, kernel_size=3, padding=1)

        # spatial refinement after FiLM
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8 if reduce_dim % 8 == 0 else 1, reduce_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, inp_image, conditional=None, return_features=False, mask=None):
        """
        Returns:
          - spatial feature map [B, reduce_dim, H, W]
          - if return_features=True: also (visual_q, cond, None)
        """
        assert isinstance(return_features, bool)

        bs = inp_image.shape[0]
        device = inp_image.device

        # --- text conditional [B,512] ---
        cond = self.get_cond_vec(conditional, bs).to(device)

        # --- base spatial features from image ---
        fmap = self.image_proj(inp_image)  # [B, C=reduce_dim, H, W]

        # --- FiLM-style modulation with text conditional ---
        if self.reduce_cond is not None:
            cond_for_film = self.reduce_cond(cond)
        else:
            cond_for_film = cond

        gamma = self.film_mul(cond_for_film)  # [B, C]
        beta = self.film_add(cond_for_film)   # [B, C]
        gamma = gamma.view(bs, -1, 1, 1)
        beta = beta.view(bs, -1, 1, 1)

        fmap = gamma * fmap + beta

        # --- spatial refinement ---
        fmap = self.spatial_refine(fmap)

        # Ensure same spatial size as input (usually already true)
        if fmap.shape[2:] != inp_image.shape[2:]:
            fmap = F.interpolate(fmap, size=inp_image.shape[2:], mode="bilinear", align_corners=True)

        # For training mix branch: provide a visual embedding from CLIP image encoder
        with torch.no_grad():
            try:
                visual_q = self.clip_model.encode_image(inp_image)
            except Exception:
                visual_q = None

        if return_features:
            return fmap, visual_q, cond, None
        return fmap


# -------------------------------------------------------------------------
# Fusion Residual Block (used in fusion head)
# -------------------------------------------------------------------------
class FusionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        assert in_channels % num_groups == 0
        assert out_channels % num_groups == 0

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.skip = (
            nn.Identity()
            if out_channels == in_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


# -------------------------------------------------------------------------
# Local decoder blocks (U-Net style) for the LFE branch
# -------------------------------------------------------------------------
class LocalDecoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x


class LocalDecoderUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = LocalDecoderConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)


# -------------------------------------------------------------------------
# Semantic Feature Adjustment (SFA) module
# -------------------------------------------------------------------------
class SFA(nn.Module):
    """
    Semantic Feature Adjustment (SFA) module.

    FiLM-style modulation:
        F_out = gamma(e) * F + beta(e)  (+ residual),
    where e is a text/prompt embedding.
    """

    def __init__(self, feat_channels: int, text_dim: int = 512, hidden: int = None, residual: bool = True):
        super().__init__()
        self.residual = residual
        h = hidden or max(256, feat_channels)
        self.proj = nn.Sequential(
            nn.Linear(text_dim, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, 2 * feat_channels),
        )

    def forward(self, fmap: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        b, c, h, w = fmap.shape
        z = self.proj(e)  # [B, 2C]
        gamma, beta = torch.chunk(z, 2, dim=1)
        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        out = gamma * fmap + beta
        return out + fmap if self.residual else out


# -------------------------------------------------------------------------
# Local Feature Extractor (LFE) with MHSA + SFA
# -------------------------------------------------------------------------
class LocalFeatureExtractor(nn.Module):
    """
    Local Feature Extractor (LFE).

    ResNet50 encoder; MHSA at enc4 & bottleneck; SFA at enc4, dec4, dec3, dec2;
    U-Net-style decoder → 32 feature channels.
    """

    def __init__(
        self,
        pretrained=True,
        num_heads=8,
        clip_ckpt="/home/mallinar/.cache/clip/ViT-B-16/ViT-B-16.pt",
    ):
        super().__init__()

        # Encoder (ResNet50)
        base = resnet50(weights=None)
        state_dict = torch.load(
            "/home/mallinar/pretrained_models/resnet50-0676ba61.pth",
            map_location="cpu",
        )
        base.load_state_dict(state_dict)
        layers = list(base.children())
        self.enc1 = nn.Sequential(*layers[:3])   # conv1+bn1+relu
        self.enc2 = nn.Sequential(*layers[3:5])  # maxpool + layer1
        self.enc3 = layers[5]                    # layer2
        self.enc4 = layers[6]                    # layer3 (1024 ch)
        self.enc5 = layers[7]                    # layer4 (2048 ch)

        # MHSA
        self.attention4 = nn.TransformerEncoderLayer(
            d_model=1024, nhead=num_heads, dim_feedforward=4096, batch_first=True
        )
        self.center_attention = nn.TransformerEncoderLayer(
            d_model=2048, nhead=num_heads, dim_feedforward=8192, batch_first=True
        )

        # Decoder
        self.up4 = LocalDecoderUpBlock(2048, 1024)
        self.up3 = LocalDecoderUpBlock(1024, 512)
        self.up2 = LocalDecoderUpBlock(512, 256)
        self.up1 = LocalDecoderUpBlock(256, 64)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # Frozen CLIP text encoder for local prompt (offline)
        if not os.path.isfile(clip_ckpt):
            raise FileNotFoundError(
                f"CLIP checkpoint for LFE not found at {clip_ckpt}. "
                f"Please copy ViT-B-16.pt there or update clip_ckpt."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = torch.jit.load(clip_ckpt, map_location=device).eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        # SFA (FiLM) sites
        self.sfa_enc4 = SFA(1024, text_dim=512)
        self.sfa_dec4 = SFA(1024, text_dim=512)
        self.sfa_dec3 = SFA(512, text_dim=512)
        self.sfa_dec2 = SFA(256, text_dim=512)

    @torch.no_grad()
    def _encode_phrase(self, phrases, device):
        if isinstance(phrases, str):
            phrases = [phrases]
        tokens = clip.tokenize(phrases).to(device)
        return self.clip_model.encode_text(tokens)  # [B', 512]

    def forward(self, x: torch.Tensor, phrase):
        b, _, h, w = x.shape
        device = x.device

        # Local text prompt embedding
        e_l = self._encode_phrase(phrase, device)  # [B', 512]
        if e_l.size(0) == 1 and b > 1:
            e_l = e_l.repeat(b, 1)
        assert e_l.size(0) == b, f"phrase batch {e_l.size(0)} != image batch {b}"

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        enc4 = self.enc4(enc3)  # [B, 1024, H4, W4]
        b4, c4, h4, w4 = enc4.shape
        enc4_seq = enc4.view(b4, c4, h4 * w4).permute(0, 2, 1)  # [B, N, C]
        enc4_seq = self.attention4(enc4_seq)
        enc4 = enc4_seq.permute(0, 2, 1).view(b4, c4, h4, w4)
        enc4 = self.sfa_enc4(enc4, e_l)

        enc5 = self.enc5(enc4)  # [B, 2048, H5, W5]
        b5, c5, h5, w5 = enc5.shape
        ctr_seq = enc5.view(b5, c5, h5 * w5).permute(0, 2, 1)
        ctr_seq = self.center_attention(ctr_seq)
        center = ctr_seq.permute(0, 2, 1).view(b5, c5, h5, w5)

        # Decoder with SFA
        dec4 = self.up4(center, enc4)
        dec4 = self.sfa_dec4(dec4, e_l)

        dec3 = self.up3(dec4, enc3)
        dec3 = self.sfa_dec3(dec3, e_l)

        dec2 = self.up2(dec3, enc2)
        dec2 = self.sfa_dec2(dec2, e_l)

        dec1 = self.up1(dec2, enc1)
        fl = self.final_up(dec1)  # [B, 32, H, W]
        if fl.shape[2:] != (h, w):
            fl = F.interpolate(fl, size=(h, w), mode="bilinear", align_corners=True)
        return fl


# -------------------------------------------------------------------------
# XBusNet full model (GFE + LFE + Fusion Head)
# -------------------------------------------------------------------------
class XBusNet(nn.Module):
    """
    XBusNet: Dual-prompt, dual-branch breast ultrasound segmentation network.
    Uses:
      - GlobalFeatureExtractor (CLIP-based, text/visual prompt)
      - LocalFeatureExtractor (ResNet50 + MHSA + SFA with local phrase)
      - Fusion head (PH) to produce final segmentation mask.
    """

    def __init__(
        self,
        size=352,
        version="ViT-B/16",
        extract_layers=(3, 7, 9),
        cond_layer=0,
        reduce_dim=64,
        n_heads=4,
        prompt="fixed",
        extra_blocks=2,
        reduce_cond=None,
        fix_shift=False,
        learn_trans_conv_only=False,
        limit_to_clip_only=False,
        upsample=False,
        add_calibration=False,
        rev_activations=False,
        trans_conv=None,
        n_tokens=None,
        complex_trans_conv=False,
        lfe_num_heads=8,
        gfe_clip_ckpt="/home/mallinar/.cache/clip/ViT-B-16/ViT-B-16.pt",
        lfe_clip_ckpt="/home/mallinar/.cache/clip/ViT-B-16/ViT-B-16.pt",
    ):
        super().__init__()

        # Global Feature Extractor (GFE)
        self.GFE = GlobalFeatureExtractor(
            version=version,
            extract_layers=extract_layers,
            cond_layer=cond_layer,
            reduce_dim=reduce_dim,
            n_heads=n_heads,
            prompt=prompt,
            extra_blocks=extra_blocks,
            reduce_cond=reduce_cond,
            fix_shift=fix_shift,
            learn_trans_conv_only=learn_trans_conv_only,
            limit_to_clip_only=limit_to_clip_only,
            upsample=upsample,
            add_calibration=add_calibration,
            rev_activations=rev_activations,
            trans_conv=trans_conv,
            n_tokens=n_tokens,
            complex_trans_conv=complex_trans_conv,
            clip_ckpt_path=gfe_clip_ckpt,
        )

        # Local Feature Extractor (LFE)
        self.LFE = LocalFeatureExtractor(
            num_heads=lfe_num_heads,
            clip_ckpt=lfe_clip_ckpt,
        )

        # Fusion Head (PH): 64 (GFE) + 32 (LFE) = 96 input channels
        self.PH = nn.Sequential(
            FusionResidualBlock(96, 64, num_groups=8),
            FusionResidualBlock(64, 64, num_groups=8),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.up_tosize = nn.Upsample(size=size, mode="bilinear", align_corners=True)

    def forward(self, x, conditional=None, phrase=None, return_features=False, **kwargs):
        """
        Args:
          x:           [B,3,H,W] input image
          conditional: global prompt (string / list[str] / tensor) – passed to GFE
          phrase:      local phrase (string / list[str]) – passed to LFE
          return_features: if True, also returns (g_feat, l_feat)

        Returns:
          seg:         [B,1,H_out,W_out] segmentation logits
        """
        # Global branch
        g_feat = self.GFE(x, conditional=conditional, return_features=False)  # [B,64,H,W]

        # Local branch
        l_feat = self.LFE(x, phrase)  # [B,32,H,W]
        if l_feat.shape[2:] != g_feat.shape[2:]:
            l_feat = F.interpolate(l_feat, size=g_feat.shape[2:], mode="bilinear", align_corners=True)

        # Fusion
        fused = torch.cat([g_feat, l_feat], dim=1)  # [B,96,H,W]
        seg = self.PH(fused)                       # [B,1,H,W]
        seg = self.up_tosize(seg)                  # [B,1,size,size]

        if return_features:
            return seg, (g_feat, l_feat)
        return seg
