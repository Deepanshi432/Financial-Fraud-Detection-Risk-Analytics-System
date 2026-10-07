import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_database():
    np.random.seed(42)
    n_samples = 15000

    # Date range over last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    random_timestamps = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(n_samples)]

    # Categorical parameters
    user_ids = [f"USER_{i:04d}" for i in np.random.randint(1000, 2000, size=n_samples)]
    locations = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "London", "New York", "Dubai", "Singapore"]
    device_types = ["Mobile Android", "Mobile iOS", "Desktop Web", "Tablet"]
    merchant_categories = ["Groceries", "Electronics", "Travel & Aviation", "Online Gaming", "Crypto Exchange", "Luxury Goods", "Utilities"]

    chosen_categories = np.random.choice(merchant_categories, size=n_samples)
    chosen_locations = np.random.choice(locations, size=n_samples)
    chosen_devices = np.random.choice(device_types, size=n_samples)
    amounts = np.round(np.random.exponential(scale=1200, size=n_samples) + 50, 2)

    # Imbalanced Fraud probability rules
    fraud_prob = []
    for i in range(n_samples):
        prob = 0.01  # Base fraud rate
        if amounts[i] > 8000:
            prob += 0.20
        if chosen_categories[i] in ["Crypto Exchange", "Online Gaming", "Luxury Goods"]:
            prob += 0.10
        if random_timestamps[i].hour in [1, 2, 3, 4]:
            prob += 0.08
        fraud_prob.append(min(prob, 0.85))

    is_fraud = [1 if np.random.rand() < p else 0 for p in fraud_prob]

    df = pd.DataFrame({
        'transaction_id': [f"TXN{i:06d}" for i in range(100000, 100000 + n_samples)],
        'user_id': user_ids,
        'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in random_timestamps],
        'amount': amounts,
        'merchant_category': chosen_categories,
        'location': chosen_locations,
        'device_type': chosen_devices,
        'is_fraud': is_fraud
    })

    # Save to SQLite Database
    conn = sqlite3.connect("fraud_analytics.db")
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()

    # Save CSV copy
    df.to_csv("transactions.csv", index=False)
    
    print("✅ Database 'fraud_analytics.db' & 'transactions.csv' created successfully!")
    print(f"📊 Total Transactions: {len(df)}")
    print(f"🚨 Fraudulent Transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")

if __name__ == "__main__":
    create_database()