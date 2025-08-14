
import numpy as np
import pandas as pd

# === Imputation helpers ===

def fill_duration(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    ref_mean = (
        X[(X['Duration'] != 0) & (X['Net Sales'] != 0)]
        .groupby(['Product Name', 'Destination', 'Net Sales'])['Duration']
        .mean()
        .astype('int')
    )
    X['Duration'] = X.apply(
        lambda row: ref_mean.get((row['Product Name'], row['Destination'], row['Net Sales']), row['Duration'])
        if row['Duration'] == 0 else row['Duration'],
        axis=1
    )
    return X

def fill_netsales(X: pd.DataFrame) -> pd.DataFrame:
    """Impute Net Sales==NaN by reference of (Product Name, Destination)."""
    X = X.copy()
    ref_mean = (
        X[X['Net Sales'].notnull()]
        .drop_duplicates(['Product Name', 'Destination'])
        .set_index(['Product Name', 'Destination'])['Net Sales']
        .to_dict()
    )
    X['Net Sales'] = X.apply(
        lambda row: ref_mean.get((row['Product Name'], row['Destination']), row['Net Sales'])
        if pd.isnull(row['Net Sales']) else row['Net Sales'],
        axis=1
    )
    return X

def fill_age(X: pd.DataFrame) -> pd.DataFrame:
    """Replace Age==118 with NaN, then impute by mean grouped on (Product Name, Net Sales)."""
    X = X.copy()
    X['Age'] = X['Age'].astype(float)
    X['Age'] = X['Age'].replace(118, np.nan)
    ref_mean = (
        X[X['Net Sales'] != 0]
        .groupby(['Product Name', 'Net Sales'])['Age']
        .mean()
    )
    X['Age'] = X.apply(
        lambda row: ref_mean.get((row['Product Name'], row['Net Sales']), row['Age'])
        if pd.isna(row['Age']) else row['Age'],
        axis=1
    )
    return X

def imput_all(df: pd.DataFrame) -> pd.DataFrame:
    function = df.copy()
    function = fill_duration(function)
    function = fill_netsales(function)
    function = fill_age(function)
    return function

# === Feature engineering ===

def feature_engineer(df):
    df = df.copy()
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    df['duration_category'] = pd.cut(
        df['Duration'],
        bins=[-1, 30, 365, float('inf')],
        labels=['Short', 'Medium', 'Long']
    )
    df['agency_product'] = df['Agency'].astype(str) + "_" + df['Product Name'].astype(str)

    return df
