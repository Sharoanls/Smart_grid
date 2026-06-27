# Smart_gridSmart Grid AI Fault Analysis Dashboard
1. Project Overview
The Smart Grid AI Fault Analysis Dashboard is a real-time, interactive web application built with Streamlit. It serves as an advanced monitoring and diagnostic tool for a microgrid system containing Distributed Energy Resources (DERs). Specifically, it is modeled around a 9-bus system integrating 3 wind turbines.

The primary objective of the system is to ingest three-phase electrical telemetry (Voltage and Current) and leverage cutting-edge Machine Learning (ML) techniques to detect and classify electrical grid faults in real-time.

2. Grid Architecture & Data
2.1 The 9-Bus DER System
The dashboard simulates a grid topology fed by three independent windmills. Each windmill is connected to the grid via three distinct bus levels:

Generator Terminal (Bus 1): Low voltage output (e.g., 690 V).
Collector Bus (Bus 2): Medium voltage step-up (e.g., 33 kV).
Point of Common Coupling / PCC (Bus 3): High voltage grid connection (e.g., 132 kV).
2.2 Telemetry & Feature Engineering
The system continuously monitors three-phase voltage (V_A, V_B, V_C) and current (I_A, I_B, I_C). From these raw signals, the dashboard calculates critical electrical metrics in real-time:

Power Factor (PF): Measures the phase angle and power efficiency of the system.
Symmetrical Components: Decomposes unbalanced three-phase vectors into:
Positive Sequence (V1, I1): Represents the balanced normal operating condition.
Negative Sequence (V2, I2): Represents the unbalance caused by asymmetrical faults (e.g., Line-to-Line).
Zero Sequence (V0, I0): Represents the unbalance caused by ground faults (e.g., Line-to-Ground).
3. Machine Learning Pipeline
The project employs a robust machine learning pipeline to classify the grid's state into one of five categories:

Normal: Healthy grid operation.
LG: Single Line-to-Ground fault.
LL: Line-to-Line fault.
LLG: Double Line-to-Ground fault.
LLLG: Three-Phase-to-Ground (Symmetrical) fault.
3.1 Data Preprocessing
SMOTE (Synthetic Minority Over-sampling Technique): Because faults (like LLLG) are statistically rare compared to Normal operations, the dataset is highly imbalanced. SMOTE is used to synthetically generate minority class samples, ensuring the AI learns to detect rare catastrophic faults just as well as common anomalies.
3.2 AI Engine Descriptions
To ensure high reliability, the system trains three distinct state-of-the-art Gradient Boosted Decision Tree (GBDT) algorithms:

NOTE

Gradient Boosting is an ensemble learning technique that builds multiple decision trees sequentially. Each new tree corrects the errors made by the previous trees, resulting in a highly accurate and robust final model.

1. CatBoost (Categorical Boosting)
Developed by Yandex, CatBoost is the primary engine used in the dashboard. It is exceptionally fast at inference and handles categorical data gracefully. It uses symmetric trees, which makes it less prone to overfitting and extremely stable for real-time telemetry classification.

2. XGBoost (Extreme Gradient Boosting)
A dominant algorithm in competitive machine learning. XGBoost is highly optimized for performance and speed. It incorporates advanced regularization (L1/L2) which prevents the model from relying too heavily on any single feature (like a single phase voltage), making it highly resilient to sensor noise.

3. LightGBM (Light Gradient Boosting Machine)
Developed by Microsoft, LightGBM uses a histogram-based algorithm and a leaf-wise tree growth strategy. It is incredibly memory-efficient and trains much faster than traditional gradient boosting on large datasets.

(Note: For demonstration purposes, the models' raw accuracies have been intentionally capped at ~98% via noise injection to simulate real-world sensor inaccuracies).

4. Dashboard Features (UI/UX)
The application is divided into several interactive modules:

METRICS: A high-level executive view showing the system health, total active faults, and comparative accuracy cards for all three AI engines.
9-BUS DER: A live topological map of the three windmills and their respective buses. It features CSS-animated rotating turbines and flashing anomaly indicators when a specific bus experiences a fault.
3D VIS (Anomaly Mapping): An interactive 3D scatter plot plotting the 3-phase voltages (V_A, V_B, V_C). It visually separates healthy data clusters from critical fault clusters.
DIAGNOSTICS: Deep-dive ML metrics for the CatBoost engine, featuring an interactive Confusion Matrix and Multi-Class ROC Curves to evaluate False Positive / True Positive rates.
AI COMPARE: A head-to-head performance comparison of CatBoost, XGBoost, and LightGBM, showing tabular results and individual 3D prediction scatter plots.
TELEMETRY: A raw data viewer containing the tabular history of the grid's electrical parameters.
TIP

This architecture is fully prepared for cloud deployment. By hosting this on Streamlit Community Cloud, grid operators can monitor the health of the 9-bus system from any web browser globally.
