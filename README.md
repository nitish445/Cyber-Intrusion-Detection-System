# Cyber Intrusion Detection System  
## Insider Threat Detection using LSTM Autoencoder and Graph Neural Network

---

## 1. Introduction

Insider threats represent one of the most challenging problems in cybersecurity, as malicious activities performed by legitimate users often resemble normal behavior. Unlike external attacks, insider threats are difficult to detect using rule-based or signature-based intrusion detection systems.

This project presents an **email-based insider threat detection system** that leverages deep learning techniques to identify anomalous user behavior without requiring labeled attack data. The system combines:

- **Temporal anomaly detection** using an LSTM Autoencoder  
- **Context-aware user classification** using a Graph Neural Network (GNN)

The proposed approach is unsupervised, scalable, and suitable for real-world enterprise environments where labeled insider attack data is rarely available.

---

## 2. Project Scope

### 2.1 In Scope
- Detection of abnormal behavior based on **email communication patterns**
- Identification of users with persistent anomalous activity
- Analysis of temporal and behavioral deviations

### 2.2 Out of Scope
The following features are **not used** because they are **not present in the dataset**:
- Login and logout times
- USB insertion events
- File system access logs
- Failed login attempts

These features commonly appear in generic intrusion detection literature but require additional log sources beyond email data.

---

## 3. Dataset Description

### 3.1 Dataset Used

This project uses the **CERT Insider Threat – Email Logs** dataset.

The dataset contains raw email communication logs with the following columns:
id, date, user, pc, to, cc, bcc, from, size, attachment, content


These fields represent **low-level system logs**, not direct machine learning features.

---

## 4. Feature Engineering

Raw log data is transformed into higher-level **behavioral features** suitable for machine learning.

### 4.1 Derived Behavioral Features

The following features are computed from the email logs:

| Feature Name | Description |
|-------------|------------|
| emails_sent | Number of emails sent by a user per day |
| unique_receivers | Number of unique recipients contacted per day |
| avg_size | Average email size (proxy for data transfer volume) |
| avg_content_len | Average email content length |
| day | Aggregated date used for temporal modeling |

Features such as `login_time`, `usb_inserted`, and `failed_logins` are **not derived** and **not approximated**, as the required data is not available in the email logs.

---

## 5. System Architecture

The system follows a multi-stage pipeline:
Email Logs (CSV)
        |
        v
Chunk-based Preprocessing
        |
        v
Daily User-Level Aggregation
        |
        v
LSTM Autoencoder
        |
        v
Reconstruction Error (Anomaly Scores)
        |
        v
User-Level Aggregation
        |
        v
Graph Neural Network
        |
        v
Suspicious Insider Identification


---

## 6. Model Description

### 6.1 LSTM Autoencoder

The LSTM Autoencoder is trained in an unsupervised manner to learn normal patterns of email communication behavior over time.

- Input: Sequences of daily behavioral features
- Objective: Minimize reconstruction error
- Output: Anomaly score per time window

High reconstruction error indicates deviation from learned normal behavior.

---

### 6.2 Graph Neural Network (GNN)

The GNN operates at the user level to classify users based on aggregated anomaly behavior.

- Nodes represent users
- Self-loop graph structure is used to preserve node-specific information
- Node labels are derived from anomaly ratios

The GNN refines temporal anomaly detection into stable user-level predictions.

---

## 7. Anomaly Detection and Labeling

The dataset does not contain ground-truth labels indicating malicious activity.

The label `is_malicious` is **predicted by the system**, not provided by the dataset:

- LSTM Autoencoder identifies anomalous behavior windows
- Anomaly ratios are computed per user
- Users exceeding a defined threshold are labeled as suspicious

---

## 8. Web-Based Interface

A web-based interface is implemented using Streamlit to demonstrate the system.

### Functionality:
- Upload a CSV file
- Automatically preprocess the data
- Run anomaly detection and classification
- Display suspicious users

The application can be launched using:
streamlit run app.py

---

## 9. Project Structure
Cyber-IDS/
│
├── app.py                  # Web application
│
├── data/                   # Raw datasets (ignored in Git)
├── processed/              # Processed data (ignored)
├── models/                 # Trained models (ignored)
│
├── train/
│   ├── preprocess.py       # Data preprocessing
│   ├── train_lstm.py       # LSTM training and anomaly detection
│   └── train_gnn.py        # GNN training
│
├── utils/
│   └── graph_utils.py      # Graph construction utilities
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 10. How to Run the Project
Step 1: Create a virtual environment
python -m venv insider_env
source insider_env/bin/activate

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Preprocess the dataset
python train/preprocess.py

Step 4: Train the LSTM Autoencoder and detect anomalies
python train/train_lstm.py

Step 5: Train the Graph Neural Network
python -u train/train_gnn.py

---

## 11. Verification and Validation
A low percentage of anomalies confirms stable LSTM learning:
1) GNN identifies a limited subset of suspicious users
2) User-level anomaly ratios provide interpretability
3) These checks validate the correctness of the pipeline.

---

## 12. Limitations
1) Only email logs are used
2) No labeled ground truth for attacks
3) Results depend on feature quality and aggregation

---

## 13. Future Work
1) Integration of additional CERT log sources (login, device, file access)
2) Attention-based sequence models
3) Explainability using attention mechanisms or SHAP
4) Deployment on cloud platforms# Cyber-Intrusion-Detection-System
