import numpy as np


def create_sequences(df, features, seq_len=7):
    sequences = []

    for user in df["user"].unique():
        user_df = df[df["user"] == user].sort_values("day")
        data = user_df[features].values

        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])

    return np.array(sequences)
