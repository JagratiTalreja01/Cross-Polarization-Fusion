from pathlib import Path

# Paths (matches your machine)
PROJECT_DIR = Path.home() / "Desktop" / "Jagrati" / "IGARSS"
DATA_DIR    = Path.home() / "Desktop" / "Jagrati" / "DeepFlood"

VV_DIR   = DATA_DIR / "SAR_VV"
VH_DIR   = DATA_DIR / "SAR_VH"
MASK_DIR = DATA_DIR / "MASK"   # must exist and filenames must match VV/VH

OUT_DIR = PROJECT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Label convention
LABEL_NONFLOOD = 0
LABEL_FLOOD    = 1
LABEL_IGNORE   = 2
