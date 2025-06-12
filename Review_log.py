import pandas as pd
import os
from datetime import datetime

def save_review_to_csv(result, csv_path="restaurant_review_analysis.csv"):
    # Add timestamp to result
    result["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert to DataFrame
    df = pd.DataFrame([result])

    # Append to file (create if it doesn't exist)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV created and review saved.")
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"✅ Review appended to existing CSV.")
