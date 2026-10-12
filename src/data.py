from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def build_preprocessor(cat_strategy="constant"):builf
    num = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    cat_imputer = (SimpleImputer(strategy="constant",fill_value="Missing")
                    if cat_strategy == "constant"
                    else SimpleImputer(strategy="most_frequent"))
    cat = make_pipeline(
        cat_imputer,
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer([
        ("num", num, make_column_selector(dtype_include="number")),
        ("cat", cat, make_column_selector(dtype_exclude="number")),
    ])