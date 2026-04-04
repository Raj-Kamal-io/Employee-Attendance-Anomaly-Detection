# Employee Attendance Anomaly Detection

An unsupervised machine learning system to detect abnormal employee attendance patterns using Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM algorithms.

---

## 📌 Project Overview

Employee attendance anomalies — such as habitual late arrivals, unexpected absences, or irregular attendance frequency — can be difficult to detect manually at scale. This project builds an unsupervised ML pipeline that automatically flags unusual attendance behavior without requiring labelled training data.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn, NumPy, Pandas
- **Algorithms:** Isolation Forest, Local Outlier Factor (LOF), One-Class SVM
- **Tools:** Jupyter Notebook, VS Code, Git

---

## ⚙️ How It Works

1. **Data Collection** — Raw employee attendance records are loaded (date, time, employee ID, status).
2. **Preprocessing** — Data is cleaned, attendance-based features are engineered, and input is normalized using standard scaling.
3. **Model Training** — Three unsupervised anomaly detection algorithms are trained on the processed data:
   - **Isolation Forest** — isolates anomalies by randomly partitioning data
   - **Local Outlier Factor (LOF)** — detects outliers based on local density deviation
   - **One-Class SVM** — learns a boundary around normal data points
4. **Anomaly Detection** — Each model flags data points that deviate significantly from normal attendance patterns.
5. **Output** — Anomalous records are identified and reported for further review.

---

## 📂 Project Structure

```
Employee-Attendance-Anomaly-Detection/
│
├── data/
│   └── attendance_data.csv       # Sample attendance dataset
│
├── notebooks/
│   └── anomaly_detection.ipynb   # Main Jupyter Notebook
│
├── src/
│   ├── preprocess.py             # Data cleaning and feature engineering
│   ├── model.py                  # ML model training and prediction
│   └── evaluate.py               # Model evaluation and comparison
│
├── requirements.txt              # Required Python packages
└── README.md
```

---

---

## 📊 Sample Output

The model outputs a flagged dataset where each record is marked as:
- **Normal (1)** — regular attendance pattern
- **Anomaly (-1)** — unusual or suspicious attendance behavior

---

## 🧠 Algorithms Comparison

| Algorithm         | Approach         | Best For                        |
|------------------|------------------|---------------------------------|
| Isolation Forest  | Tree-based       | Large datasets, fast detection  |
| LOF               | Density-based    | Local outliers in clusters      |
| One-Class SVM     | Boundary-based   | High-dimensional data           |

---

## 📈 Features Engineered

- Arrival time deviation from expected time
- Absence frequency per month
- Consecutive absence streaks
- Early departure frequency

---
