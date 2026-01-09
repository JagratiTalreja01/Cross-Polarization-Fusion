import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import OUT_DIR
from datasets.dataset_deepflood_csv import DeepFloodCSVDataset
from models.cpf_unet import CPFUNet
from losses import bce_dice_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = DeepFloodCSVDataset(OUT_DIR / "train.csv", supervised=True)
    val_ds   = DeepFloodCSVDataset(OUT_DIR / "val.csv", supervised=True)

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model = CPFUNet(base_c=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val = 1e9

    for epoch in range(1, 51):
        # ----- train -----
        model.train()
        train_loss = 0.0
        for x, y, valid, _ in tqdm(train_dl, desc=f"Train {epoch:02d}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            opt.zero_grad()
            pred = model(x).squeeze(1)   # (B,H,W)
            loss = bce_dice_loss(pred, y, valid)
            loss.backward()
            opt.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        # ----- val -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, valid, _ in tqdm(val_dl, desc=f"Val   {epoch:02d}"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)

                pred = model(x).squeeze(1)
                loss = bce_dice_loss(pred, y, valid)
                val_loss += loss.item()

        val_loss /= len(val_dl)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "cpfunet_deepflood.pth")
            print("✅ Saved best CPF model: cpfunet_deepflood.pth")


if __name__ == "__main__":
    main()
