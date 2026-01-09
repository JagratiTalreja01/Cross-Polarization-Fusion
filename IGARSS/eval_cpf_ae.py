import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import OUT_DIR
from datasets.dataset_deepflood_csv import DeepFloodCSVDataset
from models.cpf_autoencoder import CPFAutoEncoderSeg


def masked_metrics(prob, y, valid, thresh=0.5, eps=1e-6):
    pred = (prob > thresh).astype(np.uint8)
    gt = (y > 0.5).astype(np.uint8)
    v = (valid > 0.5).astype(np.uint8)

    pred = pred[v == 1]
    gt = gt[v == 1]

    inter = (pred & gt).sum()
    union = (pred | gt).sum()

    dice = (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
    iou  = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = DeepFloodCSVDataset(OUT_DIR / "test.csv", supervised=True)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = CPFAutoEncoderSeg(base_c=64).to(device)
    model.load_state_dict(torch.load("cpf_autoencoder_deepflood.pth", map_location=device))
    model.eval()

    dices, ious = [], []
    with torch.no_grad():
        for x, y, valid, _ in tqdm(dl, desc="Test CPF-AE"):
            x = x.to(device, non_blocking=True)
            prob = model(x).squeeze(1).cpu().numpy()

            y = y.numpy()
            valid = valid.numpy()

            for i in range(prob.shape[0]):
                d, j = masked_metrics(prob[i], y[i], valid[i])
                dices.append(d)
                ious.append(j)

    print(f"[CPF-AE] Test Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"[CPF-AE] Test IoU : {np.mean(ious):.4f} ± {np.std(ious):.4f}")


if __name__ == "__main__":
    main()
