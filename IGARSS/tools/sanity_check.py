import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import rasterio
import numpy as np
from config import OUT_DIR

def main():
    df = pd.read_csv(OUT_DIR / "train.csv")
    row = df.iloc[0]

    print("Sample ID:", row["id"])
    print("VV:", row["vv"])
    print("VH:", row["vh"])
    print("MASK:", row["mask"])

    with rasterio.open(row["vv"]) as src:
        print("VV shape:", src.read(1).shape, "CRS:", src.crs)
    with rasterio.open(row["vh"]) as src:
        print("VH shape:", src.read(1).shape, "CRS:", src.crs)
    with rasterio.open(row["mask"]) as src:
        m = src.read(1)
        u, c = np.unique(m, return_counts=True)
        pairs = list(zip(u.tolist(), c.tolist()))
        print("MASK value counts:", pairs)

if __name__ == "__main__":
    main()
