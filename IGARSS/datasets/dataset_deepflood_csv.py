import torch
import numpy as np
import pandas as pd
import rasterio
from torch.utils.data import Dataset

from config import LABEL_FLOOD, LABEL_IGNORE

class DeepFloodCSVDataset(Dataset):
    """
    Returns:
      - supervised=True: (x, y, valid_mask, id)
        x: float32 tensor (2,H,W) in [0,1]
        y: float32 tensor (H,W) with {0,1}
        valid_mask: float32 tensor (H,W) with {0,1} (1=valid pixel, 0=ignore)
      - supervised=False: (x, id)
    """
    def __init__(self, csv_path, supervised: bool = True):
        self.df = pd.read_csv(csv_path)
        self.supervised = supervised

        if self.supervised:
            if "mask" not in self.df.columns:
                raise ValueError("CSV has no 'mask' column.")
            if (self.df["mask"].astype(str).str.len() == 0).any():
                raise ValueError("Some rows have empty mask paths. For supervised=True, masks must exist.")

    def __len__(self):
        return len(self.df)

    def _read1(self, path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vv = self._read1(row["vv"])
        vh = self._read1(row["vh"])

        # Per-tile min-max normalization (baseline)
        vv = (vv - vv.min()) / (vv.max() - vv.min() + 1e-6)
        vh = (vh - vh.min()) / (vh.max() - vh.min() + 1e-6)

        x = np.stack([vv, vh], axis=0)  # (2,H,W)

        if self.supervised:
            mask = self._read1(row["mask"]).astype(np.int32)

            # y: flood only
            y = (mask == LABEL_FLOOD).astype(np.float32)

            # valid mask: ignore pixels where label==2
            valid = (mask != LABEL_IGNORE).astype(np.float32)

            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(valid), row["id"]

        return torch.from_numpy(x), row["id"]
