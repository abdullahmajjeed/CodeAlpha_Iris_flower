# 🪻 Iris Species Production Classification Dashboard

An end-to-end Machine Learning pipeline and dynamic dashboard built using **Python**, **Scikit-Learn**, and **Streamlit** to train a production classifier and deliver high-accuracy iris species inference outputs based on raw physical petal and sepal dimensions.

---

## 🌟 Project Overview
This repository contains a full-stack production machine learning application engineered as a submission task for the **Alpha Internship Program**. The project integrates explicit data preprocessing, Scikit-Learn Pipeline architectures, and a stateful in-memory web user interface.

It automates data pipeline staging, classifier fitting, and performance metric evaluation into a structured, highly responsive analytical web tool.

---

## 🛠️ Tech Stack & Libraries
- **Core Engine:** Python 3.x
- **Machine Learning Matrix:** Scikit-Learn (Pipelines, Estimators, Transformers)
- **Web Interface:** Streamlit (Native stateful UI microservice framework)
- **Data Structuring:** Pandas, NumPy
- **Statistical Analytics & Charts:** Seaborn, Matplotlib

---

## 📌 Architectural Features & Production Layout

The classification environment is built upon optimal machine learning engineering protocols:
1. **Scikit-Learn Pipeline Framework:** Implements a strict pipeline structure enclosing a `ColumnTransformer` with `passthrough` rules for raw continuous indicators, feeding directly into a robust `RandomForestClassifier` estimator.
2. **Stateful Caching Layer:** Leverages `@st.cache_resource` protocols to store the fully trained production classifier object in-memory, completely eliminating computing overhead on slider changes.
3. **Dynamic Value Binding:** Sidebar selector boundaries are dynamically synced with objective Pandas data minimum/maximum limits extracted directly from live csv records.

---

## 📁 Repository Structure
```text
├── iris.csv                      # Structural flower dimension tracking matrix
├── iris_classifier.py            # Stateful machine learning interface for Iris prediction
├── Iris_Classification.ipynb     # Model training, analysis, and evaluation sandbox notebook
└── README.md                     # Product presentation metadata
```

---

## 🚀 Local Deployment Instructions

Follow these sequential steps to initialize the production environment matrix on your local system:

### 1. Clone this Repository Nodes
```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
```

### 2. Install Package Dependencies
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

### 3. Initialize the Streamlit Server Engine
```bash
streamlit run irisflower.py
```

---

## 🎓 Internship Metadata
- **Program Resource:** Alpha Internship Submission Task
- **Domain Focus:** Machine Learning Pipeline Engineering, Predictive Analytics, and Stateful UI Prototyping
- **Developer:** Abdullah Majeed
