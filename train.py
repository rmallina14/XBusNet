import os
import sys
import math
import csv
import cv2
import inspect
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from functools import partial
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

from general_utils import TrainingLogger, get_attribute, filter_args, log, training_config_from_cli_args
import diceloss
import metrics  # uses metrics.compute_segmentation_scores

# Pin GPU if desired (can be overridden externally)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


# ------------------------ LR Schedules & Plots ------------------------


def cosine_warmup_lr(i, warmup=10, max_iter=90):
    if i < warmup:
        return (i + 1) / (warmup + 1)
    else:
        return 0.5 * (1.0 + math.cos(math.pi * ((i - warmup) / (max_iter - warmup))))


def save_loss_plot(train_losses, val_losses, val_interval, filename):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    if val_losses and val_interval:
        val_x = list(range(val_interval - 1, val_interval * len(val_losses), val_interval))
        plt.plot(val_x, val_losses, label="Validation Loss", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(filename)
    plt.close()


# ------------------------ Validation (metrics + optional Grad-CAM) ------------------------


def validate(model, dataset, config):
    """
    Returns:
      mean_loss: float
      val_rows:  List[[image_name, loss, iou, dice, f1, fpr, fnr]]
      agg_tuple: (mean_iou, mean_dice, mean_f1, mean_fpr, mean_fnr)

    Assumes:
      metrics.compute_segmentation_scores(pred, target)
      -> (acc, iou, dice, f1, fpr, fnr)
    """
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    # AMP handling (match training)
    use_amp = getattr(config, "amp", False)
    autocast_fn = autocast if use_amp else nullcontext

    # paths
    image_root = getattr(config, "image_dir", "images")
    run_dir = os.path.join("logs_busi", config.name)
    image_save_dir = os.path.join(run_dir, "validation_images")
    combined_save_dir = os.path.join(run_dir, "combined_visualizations")

    # Grad-CAM roots (if you later plug back CAM code)
    cam_root = os.path.join(run_dir, "gradcam")
    dec2_dir = os.path.join(cam_root, "ph_rb2")

    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(combined_save_dir, exist_ok=True)
    os.makedirs(cam_root, exist_ok=True)
    os.makedirs(dec2_dir, exist_ok=True)

    # loader + loss
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=1)
    Dice_loss = diceloss.SoftDiceLoss()
    model.eval().cuda()

    losses, ious, dices, f1s, fprs, fnrs = [], [], [], [], [], []
    val_rows = []

    # -------- pass 1: metrics & visualizations --------
    with torch.no_grad():
        for batch_idx, (data_x, data_y) in enumerate(tqdm(data_loader, desc="Validation")):
            try:
                # Dataset returns: (img, global_prompt, local_prompt, vis_image, vis_flag)
                images = data_x[0].cuda(non_blocking=True)  # [B,3,H,W]
                global_prompt = data_x[1]  # → conditional (string/list/tensor)
                local_prompt = data_x[2]  # → phrase
                masks = data_y[0].cuda(non_blocking=True)  # [B,1,H,W]

                # run model under same AMP setting as training
                with autocast_fn():
                    preds = model(
                        images,
                        conditional=global_prompt,
                        phrase=local_prompt,
                        return_features=False,
                    )
                    loss = Dice_loss(preds, masks)

                losses.append(loss.item())

                for i in range(preds.shape[0]):
                    idx = data_y[2][i].item() if len(data_y) > 2 else batch_idx * config.batch_size + i
                    image_name = dataset.image_names[idx]

                    prob_i = torch.sigmoid(preds[i].unsqueeze(0))  # [1,1,H,W]
                    mask_i = masks[i].unsqueeze(0)  # [1,1,H,W]

                    # --- metrics: use canonical 6-tuple API ---
                    acc, iou, dice, f1, fpr, fnr = metrics.compute_segmentation_scores(
                        prob_i.cpu().float(), mask_i.cpu().long()
                    )

                    ious.append(float(iou))
                    dices.append(float(dice))
                    f1s.append(float(f1))
                    fprs.append(float(fpr))
                    fnrs.append(float(fnr))

                    # --- visualizations (orig, GT, pred) ---
                    original_img_path = os.path.join(image_root, image_name)
                    orig_img = cv2.imread(original_img_path)
                    if orig_img is None:
                        raise FileNotFoundError(f"Image not found at {original_img_path}")
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    orig_h, orig_w = orig_img.shape[:2]

                    gt_mask_u8 = (mask_i[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
                    pred_bin_u8 = (prob_i[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
                    gt_mask_u8 = cv2.resize(gt_mask_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    pred_bin_u8 = cv2.resize(pred_bin_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                    combined = np.zeros((orig_h, orig_w * 3, 3), dtype=np.uint8)
                    combined[:, :orig_w] = orig_img
                    combined[:, orig_w : 2 * orig_w] = cv2.cvtColor(gt_mask_u8, cv2.COLOR_GRAY2RGB)
                    combined[:, 2 * orig_w :] = cv2.cvtColor(pred_bin_u8, cv2.COLOR_GRAY2RGB)

                    cv2.line(combined, (orig_w, 0), (orig_w, orig_h), (255, 255, 255), 2)
                    cv2.line(combined, (2 * orig_w, 0), (2 * orig_w, orig_h), (255, 255, 255), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(combined, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
                    cv2.putText(combined, "GT Mask", (orig_w + 10, 30), font, 0.8, (255, 255, 255), 2)
                    cv2.putText(combined, "Pred Mask", (2 * orig_w + 10, 30), font, 0.8, (255, 255, 255), 2)

                    Image.fromarray(pred_bin_u8).save(
                        os.path.join(image_save_dir, f"{os.path.splitext(image_name)[0]}.png")
                    )
                    cv2.imwrite(
                        os.path.join(combined_save_dir, f"{os.path.splitext(image_name)[0]}.png"),
                        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
                    )

                    val_rows.append(
                        [
                            image_name,
                            f"{loss.item():.4f}",
                            f"{iou:.4f}",
                            f"{dice:.4f}",
                            f"{f1:.4f}",
                            f"{fpr:.4f}",
                            f"{fnr:.4f}",
                        ]
                    )

            except Exception as e:
                bad_name = image_name if "image_name" in locals() else f"batch{batch_idx}"
                print(f"[Validation] Error on {bad_name}: {str(e)}")
                continue

    if not losses:
        print("Warning: No validation samples were processed successfully")
        return float("inf"), [], (0.0, 0.0, 0.0, 0.0, 0.0)

    mean_loss = float(np.mean(losses))
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
    mean_fpr = float(np.mean(fprs)) if fprs else 0.0
    mean_fnr = float(np.mean(fnrs)) if fnrs else 0.0

    print(
        f"\nValidation Complete:"
        f"\nMean Loss: {mean_loss:.4f}"
        f"\nMean IoU: {mean_iou:.4f}"
        f"\nMean Dice: {mean_dice:.4f}"
        f"\nMean F1: {mean_f1:.4f}"
        f"\nMean FPR: {mean_fpr:.4f}"
        f"\nMean FNR: {mean_fnr:.4f}"
    )

    return mean_loss, val_rows, (mean_iou, mean_dice, mean_f1, mean_fpr, mean_fnr)


# ------------------------ Main Training Loop ------------------------


def main():
    config = training_config_from_cli_args()
    val_interval = config.val_interval

    # ----- model -----
    model_cls = get_attribute(config.model)
    _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)
    model = model_cls(**model_args).cuda()

    # ----- dataset (force-set required args after filter_args) -----
    dataset_cls = get_attribute(config.dataset)
    _, dataset_args, _ = filter_args(config, inspect.signature(dataset_cls).parameters)

    def _get_cfg(name, default=None):
        return getattr(config, name, os.environ.get(name.upper(), default))

    dataset_args["image_dir"] = dataset_args.get(
        "image_dir",
        _get_cfg("image_dir", os.path.join(os.getcwd(), "images")),
    )
    dataset_args["mask_dir"] = dataset_args.get(
        "mask_dir",
        _get_cfg("mask_dir", os.path.join(os.getcwd(), "masks")),
    )
    dataset_args["csv_file"] = dataset_args.get("csv_file", _get_cfg("csv_path"))
    print("[train] dataset_args:", dataset_args)

    dataset = dataset_cls(**dataset_args)
    log.info(f"Train dataset {dataset.__class__.__name__} (length: {len(dataset)})")

    # ----- val dataset -----
    dataset_val = None
    if val_interval is not None:
        # if config behaves like a dict; otherwise adapt to your Config class
        val_dataset_args_raw = {
            k[4:]: v for k, v in config.items() if k.startswith("val_") and k != "val_interval"
        }
        _, val_dataset_args, _ = filter_args(val_dataset_args_raw, inspect.signature(dataset_cls).parameters)

        val_dataset_args["image_dir"] = val_dataset_args.get("image_dir", dataset_args["image_dir"])
        val_dataset_args["mask_dir"] = val_dataset_args.get("mask_dir", dataset_args["mask_dir"])
        val_dataset_args["csv_file"] = val_dataset_args.get("csv_file", getattr(config, "val_csv_path", None))
        val_dataset_args["split"] = val_dataset_args.get("split", "val")
        val_dataset_args["aug"] = val_dataset_args.get("aug", 0)

        print("[train] val_dataset_args:", val_dataset_args)
        dataset_val = dataset_cls(**val_dataset_args)

    # ----- optimizer & scheduler -----
    opt_cls = get_attribute(config.optimizer)
    opt = opt_cls(model.parameters(), lr=config.lr)

    if config.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, config.T_max, config.eta_min
        )
    elif config.lr_scheduler == "warmup_cosine":
        lr_scheduler = LambdaLR(
            opt,
            partial(
                cosine_warmup_lr,
                max_iter=config.max_iterations,
                warmup=getattr(config, "warmup", 10),
            ),
        )
    else:
        lr_scheduler = None

    # ----- losses, loader, amp -----
    Dice_loss = diceloss.SoftDiceLoss()
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)

    use_amp = getattr(config, "amp", False)
    autocast_fn = autocast if use_amp else nullcontext
    scaler = GradScaler() if use_amp else None

    train_losses, train_ious, val_losses = [], [], []

    # ----- logging -----
    run_dir = os.path.join("logs_busi", config.name)
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "iteration_metrics.csv")
    with open(csv_path, mode="w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["iteration", "loss", "iou"])

        with TrainingLogger(log_dir=config.name, model=model, config=config) as logger:
            i = 0
            while True:
                for data_x, data_y in data_loader:
                    # data_x = (img, global_prompt, local_prompt, vis_image, vis_flag)

                    # ----- mix logic (text + optional visual) -----
                    if getattr(config, "mix", False):
                        core = model.module if hasattr(model, "module") else model
                        # Prefer the global branch object (GFE) if present, otherwise use the model directly
                        global_branch = getattr(core, "GFE", core)

                        with autocast_fn():
                            # text-based global prompt
                            prompts = global_branch.sample_prompts(data_x[1])
                            text_cond = global_branch.compute_conditional(prompts)  # [B,512]

                            # optional visual condition from vis_image (data_x[3]) if vis_flag (data_x[4])
                            vis_img = data_x[3]
                            vis_ok = data_x[4]
                            visual_s_cond = text_cond.new_zeros(text_cond.size())

                            if isinstance(vis_img, torch.Tensor):
                                if vis_img.dim() == 3:
                                    vis_img = vis_img.unsqueeze(0)
                                use_vis = True
                                if isinstance(vis_ok, torch.Tensor):
                                    use_vis = bool(vis_ok.any().item())
                                if use_vis:
                                    v_feat, _, _ = global_branch.visual_forward(
                                        vis_img.cuda(non_blocking=True)
                                    )
                                    visual_s_cond = v_feat

                        # convex mixture of text and visual condition
                        text_weights = torch.rand(text_cond.size(0), 1, device=text_cond.device)
                        cond = text_cond * text_weights + visual_s_cond * (1 - text_weights)
                    else:
                        # no mix: pass global prompt straight through
                        cond = data_x[1]
                        if isinstance(cond, torch.Tensor):
                            cond = cond.cuda()

                    # ----- forward/backward -----
                    with autocast_fn():
                        # conditional = global prompt, phrase = local prompt
                        pred = model(
                            data_x[0].cuda(),
                            conditional=cond,
                            phrase=data_x[2],
                            return_features=False,
                        )
                        loss = Dice_loss(pred, data_y[0].cuda())

                        if torch.isnan(loss) or torch.isinf(loss):
                            log.warning("Invalid loss detected.")
                            sys.exit(-1)

                        preds = torch.sigmoid(pred)
                        preds_binary = (preds > 0.5).long()

                        # canonical 6-tuple API for metrics
                        acc, iou, dice, f1, fpr, fnr = metrics.compute_segmentation_scores(
                            preds_binary.cpu(), data_y[0].cpu().long()
                        )

                    opt.zero_grad()
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()

                    if lr_scheduler:
                        lr_scheduler.step()
                        if i % 25 == 0:
                            lr_val = lr_scheduler.get_last_lr()[0]
                            log.info(f"LR: {lr_val:.6f}")

                    # ----- logging per-iter -----
                    logger.iter(i=i, loss=loss)
                    train_losses.append(float(loss))
                    train_ious.append(float(iou))
                    csv_writer.writerow([i, float(loss), float(iou)])
                    f_csv.flush()
                    i += 1

                    # ----- termination & end-of-run validation -----
                    if i >= config.max_iterations:
                        logger.save_weights()

                        if val_interval and dataset_val is not None:
                            val_loss, val_results, (
                                val_iou,
                                val_dice,
                                val_f1,
                                val_fpr,
                                val_fnr,
                            ) = validate(model, dataset_val, config)

                            val_losses.append(val_loss)

                            val_csv_path = os.path.join(run_dir, f"val_details_iter{i}.csv")
                            with open(val_csv_path, mode="w", newline="") as val_csv_file:
                                val_writer = csv.writer(val_csv_file)
                                val_writer.writerow(
                                    ["image_name", "loss", "iou", "dice", "f1", "fpr", "fnr"]
                                )
                                for row in val_results:
                                    val_writer.writerow(row)

                            last_val_metrics = (val_iou, val_dice, val_f1, val_fpr, val_fnr)

                        save_loss_plot(
                            train_losses,
                            val_losses,
                            val_interval,
                            os.path.join(logger.base_path, "loss.png"),
                        )

                        print("\nTraining Complete:")
                        print(f"Mean Train Loss: {np.mean(train_losses):.4f}")
                        print(f"Mean Train IoU:  {np.mean(train_ious):.4f}")

                        if "last_val_metrics" in locals():
                            val_iou, val_dice, val_f1, val_fpr, val_fnr = last_val_metrics
                            print(f"Mean Val IoU:  {val_iou:.4f}")
                            print(f"Mean Val Dice: {val_dice:.4f}")
                            print(f"Mean Val F1:   {val_f1:.4f}")
                            print(f"Mean Val FPR:  {val_fpr:.4f}")
                            print(f"Mean Val FNR:  {val_fnr:.4f}")
                        else:
                            print("Validation was not run.")

                        sys.exit(0)


if __name__ == "__main__":
    main()
