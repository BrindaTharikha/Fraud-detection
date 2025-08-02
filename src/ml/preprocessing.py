import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path, parse_dates=['Timestamp'])
    return df

def feature_engineering(df):
    df['Transaction_Amount_Log'] = np.log1p(df['Transaction_Amount'])
    return df

def encode_categorical(df):
    categorical_cols = [
        'Transaction_Type', 'Device_Type', 'Location', 
        'Merchant_Category', 'Card_Type', 'Authentication_Method'
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def scale_features(X_train, X_test, cols_to_scale):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    return X_train_scaled, X_test_scaled, scaler

def save_feature_columns(columns, path="model/feature_columns.pkl"):
    joblib.dump(columns, path)

def preprocess_pipeline(df, target_col='Fraud_Label'):
    df = feature_engineering(df)
    df = encode_categorical(df)
    
    X = df.drop(columns=[target_col, 'Timestamp', 'Transaction_ID', 'User_ID'])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_cols = ['Transaction_Amount_Log', 'Account_Balance', 
                  'Avg_Transaction_Amount_7d', 'Transaction_Distance', 'Card_Age']

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scale_cols)

    # Save feature columns used for training
    save_feature_columns(X_train_scaled.columns.tolist())

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
