import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUT_DIR
from datasets.dataset_deepflood_csv import DeepFloodCSVDataset

# Choose ONE model import depending on what you want to visualize:
from models.cpf_autoencoder import CPFAutoEncoderSeg
# from models.cpf_unet import CPFUNet


def minmax01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def masked_dice_iou(prob: np.ndarray, y: np.ndarray, valid: np.ndarray, thr=0.5, eps=1e-6):
    pred = (prob > thr).astype(np.uint8)
    gt = (y > 0.5).astype(np.uint8)
    v = (valid > 0.5).astype(np.uint8)

    pred_v = pred[v == 1]
    gt_v = gt[v == 1]

    inter = (pred_v & gt_v).sum()
    union = (pred_v | gt_v).sum()

    dice = (2 * inter + eps) / (pred_v.sum() + gt_v.sum() + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def read_tif_1band(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def resize_to(arr_hw: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize 2D array to (H,W) using bilinear."""
    t = torch.from_numpy(arr_hw.astype(np.float32))[None, None, ...]  # 1x1xH0xW0
    t2 = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t2[0, 0].cpu().numpy()


def make_overlay_gray(base01: np.ndarray, pred01: np.ndarray, valid01: np.ndarray, alpha=0.55) -> np.ndarray:
    """base01: (H,W) in [0,1], returns RGB overlay with red prediction."""
    base01 = np.clip(base01, 0, 1)
    rgb = np.stack([base01, base01, base01], axis=-1)

    pred = (pred01 > 0.5)
    valid = (valid01 > 0.5)
    m = pred & valid

    # Blend red
    rgb[m, 0] = (1 - alpha) * rgb[m, 0] + alpha * 1.0
    rgb[m, 1] = (1 - alpha) * rgb[m, 1] + alpha * 0.0
    rgb[m, 2] = (1 - alpha) * rgb[m, 2] + alpha * 0.0
    return rgb


def make_overlay_rgb(base_rgb01: np.ndarray, pred01: np.ndarray, valid01: np.ndarray, alpha=0.55) -> np.ndarray:
    """base_rgb01: (H,W,3) in [0,1], returns RGB overlay with red prediction."""
    base = np.clip(base_rgb01, 0, 1).copy()

    pred = (pred01 > 0.5)
    valid = (valid01 > 0.5)
    m = pred & valid

    base[m, 0] = (1 - alpha) * base[m, 0] + alpha * 1.0
    base[m, 1] = (1 - alpha) * base[m, 1] + alpha * 0.0
    base[m, 2] = (1 - alpha) * base[m, 2] + alpha * 0.0
    return base


def cpf_composite(cpf_vv: np.ndarray, cpf_vh: np.ndarray) -> np.ndarray:
    """
    Create an RGB visualization for CPF(VV,VH):
      R = CPF-VV (norm)
      G = CPF-VH (norm)
      B = mean(VV,VH) (norm)
    """
    vv = minmax01(cpf_vv)
    vh = minmax01(cpf_vh)
    b = minmax01(0.5 * (vv + vh))
    return np.stack([vv, vh, b], axis=-1)


def save_rich_panel(
    out_png: Path,
    vv_raw: np.ndarray,
    vh_raw: np.ndarray,
    cpf_vv: np.ndarray,
    cpf_vh: np.ndarray,
    gt: np.ndarray,
    prob: np.ndarray,
    valid: np.ndarray,
    thr: float,
    title: str = ""
):
    """
    2x4 panel:
    Row1: VV raw | VH raw | CPF(VV,VH) | GT flood
    Row2: Prob   | Overlay on VV | Overlay on VH | Overlay on CPF
    """
    H, W = prob.shape

    # Ensure raw vv/vh match model output shape (common!)
    if vv_raw.shape != (H, W):
        vv_raw = resize_to(vv_raw, H, W)
    if vh_raw.shape != (H, W):
        vh_raw = resize_to(vh_raw, H, W)

    vv01 = minmax01(vv_raw)
    vh01 = minmax01(vh_raw)

    cpf_rgb = cpf_composite(cpf_vv, cpf_vh)

    pred = ((prob > thr) & (valid > 0.5)).astype(np.uint8)

    ov_vv = make_overlay_gray(vv01, pred, valid)
    ov_vh = make_overlay_gray(vh01, pred, valid)
    ov_cpf = make_overlay_rgb(cpf_rgb, pred, valid)

    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    axs = axs.ravel()

    axs[0].imshow(vv01, cmap="gray")
    axs[0].set_title("VV (raw, vis norm)")

    axs[1].imshow(vh01, cmap="gray")
    axs[1].set_title("VH (raw, vis norm)")

    axs[2].imshow(cpf_rgb)
    axs[2].set_title("CPF(VV,VH) composite")

    axs[3].imshow(gt, cmap="gray")
    axs[3].set_title("Flood mask (GT)")

    im = axs[4].imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0)
    axs[4].set_title("Prediction probability")
    plt.colorbar(im, ax=axs[4], fraction=0.046, pad=0.04)

    axs[5].imshow(ov_vv)
    axs[5].set_title("Overlay pred on VV")

    axs[6].imshow(ov_vh)
    axs[6].set_title("Overlay pred on VH")

    axs[7].imshow(ov_cpf)
    axs[7].set_title("Overlay pred on CPF(VV,VH)")

    for a in axs:
        a.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    # Examples:
    #   python view_results_cpf_richpanel.py
    #   python view_results_cpf_richpanel.py --n 12 --thr 0.4
    #   python view_results_cpf_richpanel.py --ckpt cpf_autoencoder_deepflood.pth
    n = 8
    seed = 42
    thr = 0.5

    csv_path = OUT_DIR / "test.csv"
    ckpt = Path("cpf_autoencoder_deepflood.pth")  # change if needed

    args = sys.argv[1:]
    if "--n" in args:
        n = int(args[args.index("--n") + 1])
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])
    if "--thr" in args:
        thr = float(args[args.index("--thr") + 1])
    if "--csv" in args:
        csv_path = Path(args[args.index("--csv") + 1])
    if "--ckpt" in args:
        ckpt = Path(args[args.index("--ckpt") + 1])

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}")

    random.seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] CKPT: {ckpt}")
    print(f"[INFO] thr: {thr}")

    # Dataset must contain vv/vh paths in ds.df, while x provides CPF(VV,VH)
    ds = DeepFloodCSVDataset(csv_path, supervised=True)

    # --- MODEL: AutoEncoder Seg ---
    model = CPFAutoEncoderSeg(base_c=64).to(device)
    # If you want CPFUNet instead, swap imports + this line:
    # model = CPFUNet(base_c=64).to(device)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[:min(n, len(ds))]

    out_vis = OUT_DIR / "vis_richpanel"
    out_vis.mkdir(parents=True, exist_ok=True)

    dices, ious = [], []
    print(f"[INFO] Saving panels to: {out_vis}")
    print("-" * 80)

    with torch.no_grad():
        for k, i in enumerate(idxs, start=1):
            x, y, valid, tile_id = ds[i]
            tile_id = str(tile_id)

            # CPF inputs from dataset tensor
            x1 = x.unsqueeze(0).to(device)
            prob = model(x1).squeeze().cpu().numpy().astype(np.float32)  # (H,W) or (1,H,W)->squeeze

            # Ensure prob shape is (H,W)
            if prob.ndim == 3:
                prob = prob[0]

            cpf_vv = x[0].cpu().numpy()
            cpf_vh = x[1].cpu().numpy()

            gt = (y.cpu().numpy() > 0.5).astype(np.uint8)
            vmask = (valid.cpu().numpy() > 0.5).astype(np.uint8)

            # Read raw VV/VH from CSV paths
            vv_path = ds.df.iloc[i]["vv"]
            vh_path = ds.df.iloc[i]["vh"]
            vv_raw = read_tif_1band(vv_path)
            vh_raw = read_tif_1band(vh_path)

            d, j = masked_dice_iou(prob, gt.astype(np.float32), vmask.astype(np.float32), thr=thr)
            dices.append(d)
            ious.append(j)

            out_png = out_vis / f"{tile_id}_richpanel.png"
            title = f"{tile_id} | Dice={d:.3f} IoU={j:.3f} | thr={thr}"
            save_rich_panel(
                out_png=out_png,
                vv_raw=vv_raw,
                vh_raw=vh_raw,
                cpf_vv=cpf_vv,
                cpf_vh=cpf_vh,
                gt=gt,
                prob=prob,
                valid=vmask,
                thr=thr,
                title=title
            )

            print(f"[{k:02d}/{len(idxs):02d}] {tile_id} -> {out_png.name}")

    print("-" * 80)
    print(f"[SUMMARY] N={len(dices)}  Mean Dice={np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"[SUMMARY] N={len(ious)}   Mean IoU ={np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"[DONE] Open PNGs in: {out_vis}")


if __name__ == "__main__":
    main()
