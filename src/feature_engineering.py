import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Feature engineering and preprocessing"""
    # Feature engineering
    if 'Economic Loss (Million $)' in df.columns:
        df['Economic Loss per Capita'] = df['Economic Loss (Million $)'] / df['Population (Million)']
    df['Year Difference'] = df['Year'] - 2019
    df.drop(columns=['Year'], inplace=True)
    df['Total Waste per Household'] = df['Total Waste (Tons)'] * (df['Household Waste (%)'] / 100)
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['Country', 'Food Category'])
    
    return df

import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_features(df, numeric_cols, scaler=None):
    """
    Normalize numeric features using a provided scaler (for inference),
    or fit a new one (for training).
    """
    if scaler is None:
        # Training mode
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df, scaler
    else:
        # Inference mode: Ensure df has all expected columns
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0  # Or a better default if you know it

        df = df[numeric_cols]  # Keep only expected numeric columns in correct order
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        return df

def get_feature_columns(df):
    """Identify feature columns after preprocessing"""
    return [col for col in df.columns if col != 'Economic Loss (Million $)']