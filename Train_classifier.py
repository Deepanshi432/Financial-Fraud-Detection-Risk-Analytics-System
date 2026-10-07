import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, recall_score, precision_score

def train_model():
    df = pd.read_csv("transactions.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_night_txn'] = df['hour'].apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)
    df['is_high_amount'] = df['amount'].apply(lambda x: 1 if x > 8000 else 0)

    # One-Hot Encoding
    categorical_cols = ['merchant_category', 'location', 'device_type']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    feature_cols = [c for c in df_encoded.columns if c not in ['transaction_id', 'user_id', 'timestamp', 'is_fraud']]
    X = df_encoded[feature_cols]
    y = df_encoded['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight={0: 1, 1: 8},
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.45).astype(int)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    print("--- Model Performance Metrics ---")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    with open("fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model_columns.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print("✅ Model 'fraud_model.pkl' & 'model_columns.pkl' saved successfully!")

if __name__ == "__main__":
    train_model()