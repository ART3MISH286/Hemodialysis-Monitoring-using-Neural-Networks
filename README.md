# GRU-Based Deep Temporal Model for Continuous Monitoring of Hemodialysis Patients

A Deep Learning Framework for Real-Time Dialysis Risk Prediction

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-> Overview

Hemodialysis patients require continuous monitoring to ensure safe toxin removal and balanced fluid extraction. Traditional dialysis monitoring systems rely on static thresholds and manual observation — often failing to detect early fluctuations that signal upcoming instability.

This repository presents an end-to-end Deep Learning pipeline built using Gated Recurrent Units (GRUs) to model temporal dependencies in patient vitals during hemodialysis.
The system predicts patient stability in real time, detects anomalies, and supports early medical intervention.

This work is based on the research paper:
“GRU-Based Deep Temporal Model for Continuous Monitoring of Hemodialysis Patients”
by CH. Mishmi Prashastha, J. Parna Sree, V. Manasvi,
Department of Artificial Intelligence & Machine Learning, CBIT Hyderabad.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-> Key Features

✔️ Multivariate Time-Series Modeling of dialysis session data
✔️ GRU-based deep learning architecture optimized for clinical signals
✔️ Real-time instability prediction and continuous risk scoring
✔️ Dimensionality reduction (PCA, t-SNE) for interpretability
✔️ Clinical dashboard-ready outputs for hospital deployment
✔️ High predictive performance:
	•	Accuracy: 84.67%
	•	AUC: 0.9169
✔️ Adaptive alert system for early detection of instability
✔️ Explainable temporal feature analysis

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-> Repository Structure

├── data/
│   ├── raw/                  # Raw multivariate dialysis records
│   ├── processed/            # Preprocessed & normalized datasets
│
├── notebooks/
│   ├── preprocessing.ipynb   # Data cleaning, normalization, encoding
│   ├── pca_tsne.ipynb        # Feature reduction & visualizations
│   ├── gru_model.ipynb       # Model training, validation & metrics
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_architecture.py
│   ├── train.py
│   ├── evaluate.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_auc_curve.png
│   ├── temporal_predictions.png
│
└── README.md                # You are here


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


-> Dataset Description

The dataset includes 5,000 hemodialysis session records, each structured as a multivariate time series. The most relevant features include:

🔹 Physiological Parameters
	•	Pre-, Intra-, Post-dialysis Blood Pressure
	•	Heart Rate
	•	Hemoglobin
	•	Creatinine
	•	Urea Levels

🔹 Treatment Parameters
	•	Fluid Removal Rate
	•	Dialysis Duration
	•	Ultrafiltration Dynamics
	•	Dialysis Adequacy (Kt/V)

🔹 Target Variable
	•	Disease Severity / Session Stability
	•	Stable
	•	Unstable

These features capture both short-term fluctuations and long-term treatment trends.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-> Methodology

1️. Data Preprocessing
	•	Missing values imputed (mean/median)
	•	Min–Max normalization
	•	One-hot encoding for categorical variables
	•	Train-test split: 80/20

2️. Feature Engineering & Dimensionality Reduction
	•	PCA to reduce noise & compress features
	•	t-SNE for high-dimensional visualization
	•	Identification of clinically meaningful latent variables

3️. Model Architecture (GRU Network)

Input → GRU Layer → GRU Layer → Dense → Dropout → Sigmoid/Softmax

	•	Stacked GRU layers capture temporal dependencies
	•	Dropout & batch normalization prevent overfitting
	•	Optimal setup found using Adam optimizer (lr = 0.001)
	•	Trained for 100 epochs, batch size 32
	•	Early Stopping enabled

4️. Evaluation Metrics

Metric	Value
Accuracy	84.67%
Precision (Unstable)	0.87
Recall (Unstable)	0.89
F1-Score (Unstable)	0.88
AUC	0.9169

The GRU outperformed traditional ML models such as SVM and Random Forest.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-> Real-Time Monitoring & Alert System

The model integrates into a real-time monitoring framework:

🔹 1. Data Ingestion Layer

Collects vitals from dialysis machines and connected sensors.

🔹 2. Deep Learning Inference Layer

Generates instability probability continuously across time steps.

🔹 3. Alert & Visualization Layer

Displays:
	•	Risk score timeline
	•	Predicted probability curve
	•	Binary classification outputs
	•	Alerts when risk > threshold

This enables proactive clinical decisions rather than reactive responses.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-> Interpretability & Clinical Insights

The GRU learned key patterns including:
	•	Blood Pressure Drop → strongest instability indicator
	•	Fluid Removal Rate per kg → linked to rapid onset events
	•	Hemoglobin levels → oxygen transport stability
	•	Kt/V → treatment adequacy
	•	Learned temporal embedding (“Risk Index”)

Visualizations include:

✔ Feature distributions
✔ Temporal risk probability curves
✔ GRU attention patterns
✔ PCA / t-SNE mappings

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-> Future Enhancements

We aim to expand this framework with:
	•	Incorporation of ECG, HRV, wearable sensors
	•	Deployment on edge devices for low-latency inference
	•	Transformer-based architectures for long-sequence modeling
	•	Personalized dialysis risk models using reinforcement learning
	•	Integration into hospital EHR systems

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-> References

The README includes all references cited in the original research paper.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-> Acknowledgments

Special thanks to:
Dr. Y.C.A Padmanabha Reddy Sir 
Department of Artificial Intelligence & Machine Learning, CBIT Hyderabad
for research support, computational resources, and academic mentorship.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-> Contact

For questions or collaborations:
📧 ch.mj.28@gmail.com
📧 parnasree1602@gmail.com
📧 manasvicbit@gmail.com
