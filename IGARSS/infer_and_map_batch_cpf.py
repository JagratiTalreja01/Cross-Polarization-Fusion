import os
import torch
import numpy as np
import rasterio
from tqdm import tqdm

from config import OUT_DIR
from datasets.dataset_deepflood_csv import DeepFloodCSVDataset
from models.cpf_unet import CPFUNet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CPFUNet(base_c=64).to(device)
    model.load_state_dict(torch.load("cpfunet_deepflood.pth", map_location=device))
    model.eval()

    ds = DeepFloodCSVDataset(OUT_DIR / "test.csv", supervised=False)

    out_maps = OUT_DIR / "maps_cpf"
    os.makedirs(out_maps, exist_ok=True)

    for i in tqdm(range(len(ds))):
        x, tile_id = ds[i]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            prob = model(x).squeeze().cpu().numpy().astype(np.float32)

        vv_path = ds.df.iloc[i]["vv"]
        with rasterio.open(vv_path) as src:
            meta = src.meta.copy()

        meta.update({"count": 1, "dtype": "float32"})
        out_tif = os.path.join(out_maps, f"{tile_id}_prob.tif")

        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(prob, 1)

    print(f"Saved CPF prediction rasters to: {out_maps}")


if __name__ == "__main__":
    main()
