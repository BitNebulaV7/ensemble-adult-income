from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
TARGET = "income"
EXPECTED_SHAPE = (32561, 15)


def load_adult(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw Adult dataset with proper NaNs and a binary target."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(
        path,
        names=COLUMNS,
        sep=",",
        skipinitialspace=True,  
        na_values="?",         
    )

    df[TARGET] = (df[TARGET] == ">50K").astype(int)

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape {df.shape}, expected {EXPECTED_SHAPE}")
    return df