import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.fillna("NONE")

    # Text → length
    df["content_len"] = df["content"].apply(len)

    categorical = ["user", "pc", "to", "cc", "bcc", "from", "attachment"]
    encoders = {}

    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    scaler = StandardScaler()
    df[["size", "content_len"]] = scaler.fit_transform(
        df[["size", "content_len"]]
    )

    features = [
        "user", "pc", "to", "cc", "bcc",
        "from", "size", "attachment", "content_len"
    ]

    return df, df[features].values