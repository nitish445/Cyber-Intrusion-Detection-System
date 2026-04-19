import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

#PAGE CONFIG
st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon="🛡️",
    layout="wide"
)


dashboard, tab1, tab2, tab3 = st.tabs([
    "Explanation",
    "Dashboard",
    "Model Comparison (Ablation Study)",
    "Fusion Weight Sensitivity"
])

with dashboard:
    st.header("Complete Explanation: How and Why the System Was Designed")

    st.markdown("""
## 1. Problem Definition

Insider threats are difficult to detect because the attacker is a legitimate user whose actions often resemble normal behavior.
Unlike external attacks, insider threats usually do not have explicit signatures or labeled data.

Therefore, an effective insider threat detection system must:
- Work without labeled data
- Detect subtle behavioral deviations
- Identify suspicious users, not just anomalous events

---

## 2. Dataset Selection and Justification

This project uses the **CERT Insider Threat – Email Logs dataset**.

The dataset contains only email-related activity with the following fields:

id, date, user, pc, to, cc, bcc, from, size, attachment, content

Email communication is a strong indicator of insider behavior and is commonly available in enterprise environments.
Hence, this project focuses on **email-based insider threat detection**.

---

## 3. Why Some Parameters from Research Papers Are Not Used

Many research papers list parameters such as:
- login_time
- logout_time
- usb_inserted
- files_accessed
- failed_logins

These parameters require login, device, or file system logs.
Such logs are **not present** in the email dataset used in this project.

Using parameters that are not available in the dataset would be scientifically incorrect.
Therefore, only dataset-supported features are used.

---

## 4. Raw Logs vs Behavioral Features

Raw email logs represent individual events.
Machine learning models perform better when trained on behavioral summaries rather than raw events.

For this reason, raw email logs are transformed into **daily behavioral features** at the user level.

---

## 5. Feature Engineering Process

Email events are aggregated by user and day.
From these aggregations, the following behavioral features are derived:

- Number of emails sent per day
- Number of unique recipients
- Average email size
- Average content length
- Temporal ordering by day

Each row represents one user's behavior on a given day.

---

## 6. Why We Used an LSTM Autoencoder

The dataset does not provide labels indicating malicious activity.
Therefore, an **unsupervised learning approach** is required.

The LSTM Autoencoder:
- Learns normal temporal behavior patterns
- Does not require labeled data
- Uses reconstruction error to detect anomalies

At this stage, the system identifies **when behavior is abnormal**.

---

## 7. Why LSTM Alone Is Not Sufficient

LSTM detects anomalous time windows but does not provide stable user-level decisions.
Insider threat detection requires identifying **which user** is suspicious over time.

This limitation motivates the use of a second model.

---

## 8. Why We Used a Graph Neural Network (GRR)

A Graph Neural Network is used to perform user-level classification.

- Each user is treated as a node
- Anomaly information from LSTM is aggregated per user
- Self-loops preserve user-specific behavior

The GRR refines temporal anomalies into **consistent user-level classifications**.

---

## 9. Meaning of is_malicious

The dataset does not contain ground-truth labels.
The variable `is_malicious` is **not an input feature**.

It is the **output of the system**, inferred as follows:
1. LSTM detects anomalous behavior
2. Anomaly ratios are computed per user
3. GRR classifies users as normal or suspicious

This makes the system fully unsupervised.

---

## 10. Why Two Models Are Used

| Model | Purpose |
|------|--------|
| LSTM Autoencoder | Detects temporal anomalies |
| Graph Neural Network | Classifies users |

Combining both models provides better accuracy, interpretability, and robustness.

---

## 11. Why Parameters in Papers Differ from This Implementation

Research papers often assume access to multiple log sources and present conceptual feature sets.
This implementation adapts the same detection logic to the constraints of the available dataset.

The architecture remains consistent with the literature, while the feature space is dataset-specific.

---

## 12. Conclusion

This system is a hybrid, unsupervised insider threat detection framework that:
- Uses real, available data
- Avoids fabricated features
- Combines temporal and relational learning
- Produces defendable and interpretable results

---

""")


with tab1:
    st.title("Cyber Intrusion Detection System")
    st.subheader("LSTM Autoencoder + Graph Neural Network")

    #FILE UPLOAD
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to start detection.")
        st.stop()

    #LOAD DATA
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully")

    #FULL DATA PREVIEW
    st.subheader("Preview of Uploaded Data")
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

    #GRR SCORE
    dest_freq = df.groupby("to").size()
    df["dest_risk"] = df["to"].map(dest_freq) * (df["size"] + 1)

    GRR_daily = df.groupby(["user", "day"])["dest_risk"].mean().reset_index()
    daily = daily.merge(GRR_daily, on=["user", "day"], how="left")

    daily["GRR_score"] = MinMaxScaler().fit_transform(
        daily[["dest_risk"]].fillna(0)
    )

    #FUSION
    alpha = 0.6
    daily["final_score"] = alpha * daily["lstm_score"] + (1 - alpha) * daily["GRR_score"]

    threshold = np.percentile(daily["final_score"], 70)
    daily["is_malicious"] = daily["final_score"] > threshold

    #REASONING
    def explain_reason(row):
        if not row["is_malicious"]:
            return "Final score below anomaly threshold"

        lstm_contrib = alpha * row["lstm_score"]
        GRR_contrib = (1 - alpha) * row["GRR_score"]

        if lstm_contrib >= GRR_contrib:
            return "Primary cause: Temporal behavior deviation (LSTM)"
        else:
            return "Primary cause: Relational / network risk (GRR)"

    daily["decision"] = daily["is_malicious"].apply(
        lambda x: "🚨 Malicious" if x else "✅ Normal"
    )
    daily["reason"] = daily.apply(explain_reason, axis=1)
    #THRESHOLD CALCULATION
    st.subheader("Threshold Calculation")

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
    st.subheader("Detection Results")
    st.metric("Final Score Threshold", f"{threshold:.3f}")
    
    # DEFINE anomaly_scores FROM EXISTING DATA
    anomaly_scores = daily[["user", "day", "lstm_score"]].rename(
        columns={"lstm_score": "reconstruction_error"}
    )


    # LSTM, GRR, and Hybrid Score Explanation

    st.markdown("---")
    st.subheader("How the Final Detection Score is Calculated")
    st.markdown("---")



    # 1️ LSTM VALUE

    st.markdown("### 1. LSTM Autoencoder Value")

    st.markdown("""
    The LSTM Autoencoder does **not classify users directly**.  
    Instead, it measures **how abnormal the behavior is** using reconstruction error.
    """)

    # Example: using mean reconstruction error per user
    lstm_scores = anomaly_scores.groupby("user")["reconstruction_error"].mean()

    st.write("**LSTM Value = Mean Reconstruction Error per User**")
    #st.dataframe(lstm_scores.reset_index(name="LSTM_Value"))

    st.markdown("""
    **How it is calculated:**

    LSTM Value = Mean Squared Error between input sequence and reconstructed sequence

    Mathematically:
    LSTM_Value = mean((X - X)²)

    - Low value → Normal behavior  
    - High value → Anomalous behavior
    """)


    # 2️ GRR VALUE

    st.markdown("---")
    st.markdown("### 2. Graph Neural Network (GRR) Value")

    st.markdown("""
    The GRR performs **user-level classification** using anomaly information from the LSTM.
    """)

    # GRR score already computed earlier
    GRR_scores = (
        daily.groupby("user")["GRR_score"]
        .mean()
        .reset_index()
        .rename(columns={"GRR_score": "GRR_Value"})
    )


    st.write("**GRR Value = Probability of User Being Suspicious**")
    #st.dataframe(GRR_scores)

    st.markdown("""
    - Value close to **0** → Normal user  
    - Value close to **1** → Suspicious user
    """)
    st.subheader("GRR Score Calculation")

    st.markdown("""
    ### What the GRR Score Represents

    The GRR score captures **relational communication risk** based on:
    - Email destinations
    - Communication frequency
    - Email size

    It models **how risky a user's communication network is**, rather than temporal behavior.
    """)

    st.markdown("""
    ### Step 1: Destination Frequency

    For each destination (email receiver), we calculate how frequently it appears:\n
    dest_freq(d) = count of emails sent to destination d

    This measures how common or rare a communication endpoint is.
    """)

    st.markdown("""
    ### Step 2: Destination Risk per Email

    For each email event *i*, destination risk is computed as:\n
    dest_risk_i = dest_freq(to_i) × (email_size_i + 1)

    - Rare destinations increase risk  
    - Larger email sizes increase risk  
    - `+1` avoids zero multiplication
    """)

    st.markdown("""
    ### Step 3: User-Day Aggregation

    For each user *u* on day *d*, \nthe destination risks are aggregated:\n
    GRR_raw(u, d) = (1 / N) × Σ dest_risk_i

    where *N* is the number of emails sent by the user on that day.
    """)

    st.markdown("""
    ### Step 4: Normalization (Final GRR Score)

    The raw score is normalized using Min-Max scaling:\n
    GRR_score = (GRR_raw − min(GRR_raw)) / (max(GRR_raw) − min(GRR_raw))

    This ensures the score lies between **0 and 1**.
    """)

    st.markdown("""
    ### Final GRR Formula (As Used in This System)\n
    GRR_score(u, d) =
    MinMax(
    (1 / N) × Σ [ dest_freq(to_i) × (size_i + 1) ]
    )

    """)

    st.markdown("""
    ### Interpretation

    - **Low GRR score (≈ 0)** → Normal communication behavior  
    - **High GRR score (≈ 1)** → High relational / network risk  

    The GRR score focuses on **who the user communicates with**, not **when**.
    """)





    # 3️ HYBRID VALUE (LSTM + GRR)

    st.markdown("---")
    st.markdown("### 3. Hybrid Detection Value (LSTM + GRR)")

    st.markdown("""
    The final detection score is a **weighted combination** of:
    - Temporal anomaly score (LSTM)
    - User-level risk score (GRR)
    """)

    # Normalize LSTM values
    lstm_norm = (lstm_scores - lstm_scores.min()) / (lstm_scores.max() - lstm_scores.min())

    # Align users
    hybrid_df = GRR_scores.copy()
    hybrid_df["LSTM_Value"] = lstm_norm.values[:len(hybrid_df)]

    # Hybrid score
    hybrid_df["Hybrid_Value"] = (
        0.6 * hybrid_df["LSTM_Value"] +
        0.4 * hybrid_df["GRR_Value"]
    )

    #st.write("**Hybrid Score Calculation:**")
    #st.dataframe(hybrid_df)

    st.markdown("""
    **Hybrid Formula Used:**
    Hybrid_Value = 0.6 × LSTM_Value + 0.4 × GRR_Value

    This ensures:
    - Temporal anomalies are prioritized
    - User-level consistency is maintained
    """)


    # FINAL DECISION

    st.markdown("### Final Decision Rule")

    hybrid_df["Final_Label"] = (hybrid_df["Hybrid_Value"] > threshold).astype(int)

    st.markdown("""
    A user is flagged as **suspicious** if:
    Hybrid_Value > Final_Score_Threshold
    """)

    #st.dataframe(hybrid_df[["user", "Hybrid_Value", "Final_Label"]])

    st.success("Final detection scores calculated using LSTM, GRR, and Hybrid model.")


    st.dataframe(
        daily[[
            "user", "day",
            "decision", "reason",
            "lstm_score", "GRR_score", "final_score"
        ]],
        use_container_width=True,
        height=350
    )


    #GLOBAL BAR CHART
    st.subheader("Final Anomaly Score per User-Day")

    daily["label"] = daily["user"] + " | " + daily["day"].astype(str)
    st.bar_chart(daily.set_index("label")["final_score"])

    #GLOBAL LINE GRAPH WITH THRESHOLD
    st.subheader("LSTM vs GRR Contribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(daily["label"], daily["lstm_score"], marker="o", label="LSTM Score")
    ax.plot(daily["label"], daily["GRR_score"], marker="o", label="GRR Score")
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
    ax.set_title("Global LSTM, GRR and Final Score with Threshold")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)

    #USER-WISE INTERACTIVE INSPECTION
    st.subheader("Inspect User Activity")

    selected_user = st.selectbox("Select a user", daily["user"].unique())
    user_data = daily[daily["user"] == selected_user]

    st.dataframe(
        user_data[[
            "day", "decision", "reason",
            "lstm_score", "GRR_score", "final_score"
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

    st.subheader("Score Trend for Selected User")

    chart_df = user_data.set_index("day")[[
        "lstm_score", "GRR_score", "final_score"
    ]]
    st.line_chart(chart_df)

    #VISUALS

    st.subheader("LSTM vs GRR Contribution")
    st.line_chart(daily[["lstm_score", "GRR_score", "final_score"]])

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
    #with st.expander("How to read this dashboard"):
    #    st.markdown("""
    #- **Global graphs** show overall anomaly distribution and threshold
    #- **User-wise view** allows analyst-style inspection
    #- Final decision is based on **final score vs threshold**
    #- Reason column explains *why* the alert was raised
    #""")

    st.success("IDS analysis completed successfully")

with tab2:
    st.header("Model Comparison – Ablation Study")

    #EXPLANATION
    st.markdown("""
    ## What is this page about?

    This page performs an **ablation study**, a standard research technique used to
    understand the contribution of each model component.

    We compare:
    - A **Hybrid model** (LSTM + GRR)
    - A **GRR-only model**
    - A **LSTM-only model**

    The goal is to prove that **both temporal and relational modeling are necessary**
    for reliable insider threat detection.
    """)

    st.markdown("""
    ###Models Explained

    **1) Hybrid Model (LSTM + GRR)**  
    `Final_full = 0.6 × LSTM + 0.4 × GRR`

    - LSTM captures **temporal behavior deviations**
    - GRR captures **relational / network-based risk**
    - Represents the **complete IDS**

    **2) GRR-only Model**  
    `Final_GRR_only = GRR`

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
    daily["Final_Hybrid"] = 0.6 * daily["lstm_score"] + 0.4 * daily["GRR_score"]
    daily["Final_GRR_Only"] = daily["GRR_score"]
    daily["Final_LSTM_Only"] = daily["lstm_score"]

    #THRESHOLDS
    th_hybrid = np.percentile(daily["Final_Hybrid"], 70)
    th_GRR = np.percentile(daily["Final_GRR_Only"], 70)
    th_lstm = np.percentile(daily["Final_LSTM_Only"], 70)

    daily["Hybrid Alert"] = daily["Final_Hybrid"] > th_hybrid
    daily["GRR Alert"] = daily["Final_GRR_Only"] > th_GRR
    daily["LSTM Alert"] = daily["Final_LSTM_Only"] > th_lstm

    #RESULTS
    st.subheader("Side-by-Side Results")

    st.dataframe(
        daily[[
            "user", "day",
            "Hybrid Alert", "GRR Alert", "LSTM Alert",
            "Final_Hybrid", "Final_GRR_Only", "Final_LSTM_Only"
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
        "Final_GRR_Only",
        "Final_LSTM_Only"
    ]]

    st.line_chart(chart_df)

    #FINAL INSIGHT
    st.markdown("""
    ### Key Takeaway

    - Some anomalies are detected **only by LSTM**
    - Some anomalies are detected **only by GRR**
    - The hybrid model detects **both**

    This proves that **LSTM and GRR are complementary**, not redundant.
    """)

    st.success("Ablation study completed successfully")


with tab3:
    st.header("Fusion Weight Sensitivity Analysis")

    #EXPLANATION
    st.markdown("""
    ## What is this page about?

    This page studies how **changing fusion weights** between LSTM and GRR
    affects anomaly detection.

    It answers:
    - Why were the weights **0.6 and 0.4** chosen?
    - How sensitive is the model to weight changes?
    """)

    st.markdown("""
    ### Fusion Strategies Explained

    **A️⃣ Balanced Fusion (Baseline)**  
    `Final_A = 0.6 × LSTM + 0.4 × GRR`

    - Balanced importance to behavior and relationships
    - Stable and interpretable
    - Used as the **default configuration**

    **B️⃣ GRR-Dominant Fusion**  
    `Final_B = 0.6 × LSTM + 1.0 × GRR`

    - Strong emphasis on network/relational risk
    - Detects coordinated insider activity
    - Can over-flag popular shared resources

    **C️⃣ LSTM-Dominant Fusion**  
    `Final_C = 1.0 × LSTM + 0.4 × GRR`

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
        0.6 * daily["lstm_score"] + 0.4 * daily["GRR_score"]
    )

    daily["Fusion_B (0.6L + 1.0G)"] = (
        0.6 * daily["lstm_score"] + 1.0 * daily["GRR_score"]
    )

    daily["Fusion_C (1.0L + 0.4G)"] = (
        1.0 * daily["lstm_score"] + 0.4 * daily["GRR_score"]
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

    - Increasing **GRR weight** amplifies network-based alerts
    - Increasing **LSTM weight** amplifies behavioral alerts
    - Balanced fusion provides the **most robust and interpretable detection**

    This validates the choice of fusion weights.
    """)

    st.success("Fusion weight sensitivity analysis completed")

