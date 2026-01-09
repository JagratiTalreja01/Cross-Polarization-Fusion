import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from config import OUT_DIR

def main():
    df = pd.read_csv(OUT_DIR / "deepflood_index.csv")

    # Ensure masks exist (supervised training)
    df = df[df["mask"].astype(str).str.len() > 0].reset_index(drop=True)

    train, temp = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    val, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)

    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    print("\nWrote splits to outputs/: train.csv, val.csv, test.csv")
    print(f"Train={len(train)} Val={len(val)} Test={len(test)}")

if __name__ == "__main__":
    main()
