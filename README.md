Credit Card Fraud Detection using AutoEncoder
This project implements an unsupervised deep learning approach for detecting fraudulent credit card transactions using the PyOD library’s AutoEncoder model. The model learns normal transaction patterns and detects anomalies based on reconstruction error.

📂 Project Structure

.
├── data/
│   └── creditcard.csv               # Dataset (download from Kaggle)
│
├── src/
│   └── fraud_detection.py            # Main training and evaluation script
│
├── output/
│   ├── confusion_matrix.png          # Classification performance heatmap
│   ├── precision_recall_curve.png    # PR curve for anomaly detection
│   └── prediction_results.csv        # Model predictions and anomaly scores
│
├── manifest.txt                      # Project description and dependencies
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Ignore virtual environment, dataset, and generated files

📊 Dataset
The dataset used is from Kaggle:
Credit Card Fraud Detection Dataset

Transactions: 284,807

Frauds: 492 (0.172%)

Features: 30 (28 anonymized PCA components + Time + Amount)

Class label: 0 = normal, 1 = fraud

🛠 Installation
Clone the repository

git clone https://github.com/Sandeep271/fraud-detection-autoencoder.git
cd fraud-detection-autoencoder
Create and activate virtual environment

python -m venv venv
venv\Scripts\activate   # Windows

Install dependencies

pip install -r requirements.txt
Download dataset
Download creditcard.csv from Kaggle and place it inside the data/ folder.

🔍 How It Works
The AutoEncoder detects fraud in four main steps:

         ┌───────────────┐
         │  Input Data   │
         └──────┬────────┘
                │
        Normalize & Scale
                │
         ┌──────▼────────┐
         │ AutoEncoder   │
         │ (Trained on   │
         │ normal data)  │
         └──────┬────────┘
                │
   Reconstruction & Error
                │
   ┌────────────▼─────────────┐
   │ High Error → Anomaly (1) │
   │ Low Error  → Normal  (0) │
   └─────────────────────────┘
Preprocessing – Data is normalized and only normal transactions are used to train the AutoEncoder.

Encoding – The network compresses the input features into a smaller latent representation.

Decoding – The network reconstructs the original data from the compressed representation.

Error Measurement – Large reconstruction errors indicate anomalies (possible fraud).

🚀 Running the Project

python src/fraud_detection.py
📈 Results
Confusion Matrix

Precision-Recall Curve

Key Metrics

Precision, Recall, F1-Score

ROC-AUC Score

PR-AUC Score

📌 Notes
The AutoEncoder is trained only on normal transactions to learn typical patterns.

Fraudulent transactions are detected as anomalies (1).

You can adjust parameters like epochs, hidden_neurons, and contamination in fraud_detection.py to tune performance.

🧑‍💻 Author
Sandeep
GitHub: Sandeep271

