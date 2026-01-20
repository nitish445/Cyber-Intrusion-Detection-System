import os

import pandas as pd

CHUNK_SIZE = 200_000
INPUT_FILE = "data/email.csv"
OUTPUT_FILE = "processed/user_daily.csv"

os.makedirs("processed", exist_ok=True)

aggregated_chunks = []

print("🚀 Starting preprocessing...")

for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)):
    print(f"Processing chunk {i+1}")

    # Convert date
    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
    chunk = chunk.dropna(subset=["date"])
    chunk["day"] = chunk["date"].dt.date

    # Feature engineering
    chunk["content_len"] = chunk["content"].fillna("").astype(str).str.len()
    chunk["has_attach"] = 0

    # Aggregate per user per day
    daily = chunk.groupby(["user", "day"]).agg(
        emails_sent=("id", "count"),
        unique_receivers=("to", "nunique"),
        avg_size=("size", "mean"),
        attachments=("has_attach", "sum"),
        avg_content_len=("content_len", "mean")
    ).reset_index()


    aggregated_chunks.append(daily)

# Combine all chunks
final_df = pd.concat(aggregated_chunks)
final_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Preprocessing completed successfully!")
print(f"Saved to {OUTPUT_FILE}")
