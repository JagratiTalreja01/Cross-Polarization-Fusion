import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import csv
from config import VV_DIR, VH_DIR, MASK_DIR, OUT_DIR

def main():
    vv_files = sorted(list(VV_DIR.glob("*.tif")) + list(VV_DIR.glob("*.tiff")))
    if not vv_files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {VV_DIR}")

    rows = []
    missing_vh = 0
    missing_mask = 0
    has_mask_dir = MASK_DIR.exists()

    for vv in vv_files:
        name = vv.name
        vh = VH_DIR / name
        if not vh.exists():
            missing_vh += 1
            continue

        mask = (MASK_DIR / name) if has_mask_dir else None
        if has_mask_dir and not mask.exists():
            missing_mask += 1
            mask = None

        rows.append({
            "id": name.rsplit(".", 1)[0],
            "vv": str(vv),
            "vh": str(vh),
            "mask": str(mask) if mask else ""
        })

    out_csv = OUT_DIR / "deepflood_index.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "vv", "vh", "mask"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_csv}")
    print(f"Pairs kept: {len(rows)}")
    print(f"Missing VH: {missing_vh}")
    print(f"Mask dir present: {has_mask_dir}")
    if has_mask_dir:
        print(f"Missing masks: {missing_mask}")

if __name__ == "__main__":
    main()
