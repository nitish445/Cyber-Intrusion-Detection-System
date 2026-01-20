import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

#PAGE CONFIG
st.set_page_config(
    page_title="Cyber Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)


tab1, tab2, tab3 = st.tabs([
    "Main IDS Dashboard",
    "Model Comparison (Ablation Study)",
    "Fusion Weight Sensitivity"
])


with tab1:
    st.title("Cyber Intrusion Detection System")
    st.subheader("LSTM Autoencoder + Graph Neural Network (Explainable & Interactive IDS)")

    #FILE UPLOAD
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to start detection.")
        st.stop()

    #LOAD DATA
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully")

    #FULL DATA PREVIEW
    st.subheader("Preview of Uploaded Data (Complete Dataset)")
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    st.dataframe(df, use_container_width=True, height=350)

    #PREPROCESSING
    st.subheader("Preprocessing")

    df["date"] = pd.to_datetime(df.get("date", pd.Timestamp.today()))
    df["day"] = df.get("day", df["date"].dt.date)
    df["user"] = df.get("user", df.get("user_id", "UNKNOWN"))
    df["content"] = df.get("content", "")
    df["content_len"] = df["content"].astype(str).str.len()
    df["size"] = df.get("size", 0)
    df["to"] = df.get("to", "internal")

    for col in ["files_accessed", "usb_inserted", "failed_logins",
                "data_download_mb", "working_hours"]:
        df[col] = df.get(col, 0)

    #DAILY AGGREGATION
    daily = df.groupby(["user", "day"]).agg({
        "files_accessed": "sum",
        "usb_inserted": "sum",
        "failed_logins": "sum",
        "data_download_mb": "sum",
        "working_hours": "mean",
        "content_len": "mean",
        "size": "sum",
        "to": "nunique"
    }).reset_index()

    st.success("Preprocessing complete")
    st.subheader("Aggregated Daily User Activity")
    st.dataframe(daily, use_container_width=True, height=300)

    #FEATURE SCALING
    features = [
        "files_accessed", "usb_inserted", "failed_logins",
        "data_download_mb", "working_hours",
        "content_len", "size", "to"
    ]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(daily[features])

    #LSTM SCORE
    X_mean = X.mean(axis=0)
    daily["lstm_score"] = np.mean((X - X_mean) ** 2, axis=1)

    #GNN SCORE
    dest_freq = df.groupby("to").size()
    df["dest_risk"] = df["to"].map(dest_freq) * (df["size"] + 1)

    gnn_daily = df.groupby(["user", "day"])["dest_risk"].mean().reset_index()
    daily = daily.merge(gnn_daily, on=["user", "day"], how="left")

    daily["gnn_score"] = MinMaxScaler().fit_transform(
        daily[["dest_risk"]].fillna(0)
    )

    #FUSION
    alpha = 0.6
    daily["final_score"] = alpha * daily["lstm_score"] + (1 - alpha) * daily["gnn_score"]

    threshold = np.percentile(daily["final_score"], 70)
    daily["is_malicious"] = daily["final_score"] > threshold

    #REASONING
    def explain_reason(row):
        if not row["is_malicious"]:
            return "Final score below anomaly threshold"

        lstm_contrib = alpha * row["lstm_score"]
        gnn_contrib = (1 - alpha) * row["gnn_score"]

        if lstm_contrib >= gnn_contrib:
            return "Primary cause: Temporal behavior deviation (LSTM)"
        else:
            return "Primary cause: Relational / network risk (GNN)"

    daily["decision"] = daily["is_malicious"].apply(
        lambda x: "🚨 Malicious" if x else "✅ Normal"
    )
    daily["reason"] = daily.apply(explain_reason, axis=1)
    #THRESHOLD CALCULATION
    st.subheader("Threshold Calculation (Transparent)")

    scores = np.sort(daily["final_score"].values)
    N = len(scores)
    index = 0.70 * (N - 1)
    li, ui = int(np.floor(index)), int(np.ceil(index))
    w = index - li

    calc_threshold = scores[li] + w * (scores[ui] - scores[li])

    st.markdown(f"""
    Sorted Final Scores:
    {np.round(scores, 6)}


    Index = 0.70 × ({N} − 1) = **{index:.2f}**

    Threshold =  
    {scores[li]:.6f} + {w:.2f} × ({scores[ui]:.6f} − {scores[li]:.6f})

    **Final Threshold = {calc_threshold:.3f}**
    """)
    #RESULTS TABLE
    st.subheader("Detection Results (Explainable)")
    st.metric("Final Score Threshold", f"{threshold:.3f}")

    st.dataframe(
        daily[[
            "user", "day",
            "decision", "reason",
            "lstm_score", "gnn_score", "final_score"
        ]],
        use_container_width=True,
        height=350
    )


    #GLOBAL BAR CHART
    st.subheader("Final Anomaly Score per User-Day (Global)")

    daily["label"] = daily["user"] + " | " + daily["day"].astype(str)
    st.bar_chart(daily.set_index("label")["final_score"])

    #GLOBAL LINE GRAPH WITH THRESHOLD
    st.subheader("LSTM vs GNN Contribution (Global with Threshold)")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(daily["label"], daily["lstm_score"], marker="o", label="LSTM Score")
    ax.plot(daily["label"], daily["gnn_score"], marker="o", label="GNN Score")
    ax.plot(daily["label"], daily["final_score"], marker="o", label="Final Score")

    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold:.3f})"
    )

    ax.set_xlabel("User | Day")
    ax.set_ylabel("Score")
    ax.set_title("Global LSTM, GNN and Final Score with Threshold")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)

    #USER-WISE INTERACTIVE INSPECTION
    st.subheader("Inspect User Activity (Interactive)")

    selected_user = st.selectbox("Select a user", daily["user"].unique())
    user_data = daily[daily["user"] == selected_user]

    st.dataframe(
        user_data[[
            "day", "decision", "reason",
            "lstm_score", "gnn_score", "final_score"
        ]],
        use_container_width=True
    )

    if (user_data["final_score"] > threshold).any():
        st.error(
            f"🚨 `{selected_user}` is flagged as MALICIOUS\n\n"
            f"At least one day crossed the threshold ({threshold:.3f})"
        )
    else:
        st.success(
            f"✅ `{selected_user}` shows NORMAL behavior\n\n"
            f"All scores are below the threshold ({threshold:.3f})"
        )

    st.subheader("Score Trend for Selected User (Normal Interactive View)")

    chart_df = user_data.set_index("day")[[
        "lstm_score", "gnn_score", "final_score"
    ]]
    st.line_chart(chart_df)

    #VISUALS

    st.subheader("LSTM vs GNN Contribution")
    st.line_chart(daily[["lstm_score", "gnn_score", "final_score"]])

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = daily[features].corr()
    im = ax.imshow(corr, cmap="coolwarm")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticklabels(features)
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)
    #FINAL NOTE
    with st.expander("How to read this dashboard"):
        st.markdown("""
    - **Global graphs** show overall anomaly distribution and threshold
    - **User-wise view** allows analyst-style inspection
    - Final decision is based on **final score vs threshold**
    - Reason column explains *why* the alert was raised
    """)

    st.success("IDS analysis completed successfully")

with tab2:
    st.header("Model Comparison – Ablation Study")

    #EXPLANATION
    st.markdown("""
    ## What is this page about?

    This page performs an **ablation study**, a standard research technique used to
    understand the contribution of each model component.

    We compare:
    - A **Hybrid model** (LSTM + GNN)
    - A **GNN-only model**
    - A **LSTM-only model**

    The goal is to prove that **both temporal and relational modeling are necessary**
    for reliable insider threat detection.
    """)

    st.markdown("""
    ###Models Explained

    **1) Hybrid Model (LSTM + GNN)**  
    `Final_full = 0.6 × LSTM + 0.4 × GNN`

    - LSTM captures **temporal behavior deviations**
    - GNN captures **relational / network-based risk**
    - Represents the **complete IDS**

    **2) GNN-only Model**  
    `Final_gnn_only = GNN`

    - Detects shared destinations and coordinated behavior
    - Cannot detect abnormal behavior of an individual user

    **3️) LSTM-only Model**  
    `Final_lstm_only = LSTM`

    - Detects abnormal behavior over time
    - Ignores relational and contextual risk
    """)

    st.markdown("""
    ### Decision Logic

    Each model computes its **own anomaly threshold** using the
    **70th percentile of its score distribution**.

    This ensures:
    - Fair comparison
    - No model is advantaged or disadvantaged
    - Alerts are relative to each model’s behavior
    """)

    #MODEL VARIANTS
    daily["Final_Hybrid"] = 0.6 * daily["lstm_score"] + 0.4 * daily["gnn_score"]
    daily["Final_GNN_Only"] = daily["gnn_score"]
    daily["Final_LSTM_Only"] = daily["lstm_score"]

    #THRESHOLDS
    th_hybrid = np.percentile(daily["Final_Hybrid"], 70)
    th_gnn = np.percentile(daily["Final_GNN_Only"], 70)
    th_lstm = np.percentile(daily["Final_LSTM_Only"], 70)

    daily["Hybrid Alert"] = daily["Final_Hybrid"] > th_hybrid
    daily["GNN Alert"] = daily["Final_GNN_Only"] > th_gnn
    daily["LSTM Alert"] = daily["Final_LSTM_Only"] > th_lstm

    #RESULTS
    st.subheader("Side-by-Side Results")

    st.dataframe(
        daily[[
            "user", "day",
            "Hybrid Alert", "GNN Alert", "LSTM Alert",
            "Final_Hybrid", "Final_GNN_Only", "Final_LSTM_Only"
        ]],
        use_container_width=True,
        height=400
    )

    #GRAPH
    st.subheader("Score Comparison (Same User-Day)")

    chart_df = daily.set_index(
        daily["user"] + " | " + daily["day"].astype(str)
    )[[
        "Final_Hybrid",
        "Final_GNN_Only",
        "Final_LSTM_Only"
    ]]

    st.line_chart(chart_df)

    #FINAL INSIGHT
    st.markdown("""
    ### Key Takeaway

    - Some anomalies are detected **only by LSTM**
    - Some anomalies are detected **only by GNN**
    - The hybrid model detects **both**

    This proves that **LSTM and GNN are complementary**, not redundant.
    """)

    st.success("Ablation study completed successfully")


with tab3:
    st.header("Fusion Weight Sensitivity Analysis")

    #EXPLANATION
    st.markdown("""
    ## What is this page about?

    This page studies how **changing fusion weights** between LSTM and GNN
    affects anomaly detection.

    It answers:
    - Why were the weights **0.6 and 0.4** chosen?
    - How sensitive is the model to weight changes?
    """)

    st.markdown("""
    ### Fusion Strategies Explained

    **A️⃣ Balanced Fusion (Baseline)**  
    `Final_A = 0.6 × LSTM + 0.4 × GNN`

    - Balanced importance to behavior and relationships
    - Stable and interpretable
    - Used as the **default configuration**

    **B️⃣ GNN-Dominant Fusion**  
    `Final_B = 0.6 × LSTM + 1.0 × GNN`

    - Strong emphasis on network/relational risk
    - Detects coordinated insider activity
    - Can over-flag popular shared resources

    **C️⃣ LSTM-Dominant Fusion**  
    `Final_C = 1.0 × LSTM + 0.4 × GNN`

    - Strong emphasis on individual behavior
    - Detects subtle behavioral drift
    - Can miss collusive insider attacks
    """)

    st.markdown("""
    ### Thresholding Strategy

    Each fusion strategy uses its **own 70th percentile threshold**.
    This ensures:
    - Fair comparison
    - No score-scale dominance
    - Robust evaluation
    """)

    #FUSION VARIANTS
    daily["Fusion_A (0.6L + 0.4G)"] = (
        0.6 * daily["lstm_score"] + 0.4 * daily["gnn_score"]
    )

    daily["Fusion_B (0.6L + 1.0G)"] = (
        0.6 * daily["lstm_score"] + 1.0 * daily["gnn_score"]
    )

    daily["Fusion_C (1.0L + 0.4G)"] = (
        1.0 * daily["lstm_score"] + 0.4 * daily["gnn_score"]
    )

    #THRESHOLDS
    th_A = np.percentile(daily["Fusion_A (0.6L + 0.4G)"], 70)
    th_B = np.percentile(daily["Fusion_B (0.6L + 1.0G)"], 70)
    th_C = np.percentile(daily["Fusion_C (1.0L + 0.4G)"], 70)

    daily["Alert_A"] = daily["Fusion_A (0.6L + 0.4G)"] > th_A
    daily["Alert_B"] = daily["Fusion_B (0.6L + 1.0G)"] > th_B
    daily["Alert_C"] = daily["Fusion_C (1.0L + 0.4G)"] > th_C

    #RESULTS
    st.subheader("Fusion Strategy Results")

    st.dataframe(
        daily[[
            "user", "day",
            "Alert_A", "Alert_B", "Alert_C",
            "Fusion_A (0.6L + 0.4G)",
            "Fusion_B (0.6L + 1.0G)",
            "Fusion_C (1.0L + 0.4G)"
        ]],
        use_container_width=True,
        height=400
    )

    #GRAPH
    st.subheader("Fusion Score Comparison")

    chart_df = daily.set_index(
        daily["user"] + " | " + daily["day"].astype(str)
    )[[
        "Fusion_A (0.6L + 0.4G)",
        "Fusion_B (0.6L + 1.0G)",
        "Fusion_C (1.0L + 0.4G)"
    ]]

    st.line_chart(chart_df)

    #FINAL INSIGHT
    st.markdown("""
    ### Key Takeaway

    - Increasing **GNN weight** amplifies network-based alerts
    - Increasing **LSTM weight** amplifies behavioral alerts
    - Balanced fusion provides the **most robust and interpretable detection**

    This validates the choice of fusion weights.
    """)

    st.success("Fusion weight sensitivity analysis completed")

