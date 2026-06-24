# 🌍 Carbon AI — Industrial Carbon Intelligence & Sustainability Analytics Platform

> Transforming industrial operational data into actionable carbon intelligence through Machine Learning, Analytics, and Sustainability Insights.

---
🚀 Live Demo

🌍 Carbon AI Dashboard
[Deploy](https://carbon-ai-industrial-carbon-intelligence-platform-mcyncfxgfv8w.streamlit.app/)

## 📌 Overview

Carbon AI is an end-to-end Data Science project designed to analyze, predict, and monitor industrial carbon emissions using Machine Learning and interactive analytics.

The platform combines industrial operational data with global CO₂ emission trends to help organizations understand emission drivers, forecast future emissions, assess carbon risk, and identify sustainability opportunities.

Unlike traditional machine learning projects that stop at prediction, Carbon AI delivers a complete workflow from data analysis to business-oriented recommendations through an interactive dashboard.

---

## 🚀 Key Features

### 🤖 Carbon Emission Prediction

Predict industrial carbon emissions using a trained Random Forest Regression model.

### 📊 Global Carbon Analytics

Analyze historical country-level CO₂ emission trends and identify major emitters.

### 📈 Future Emission Forecasting

Forecast future carbon emissions using historical environmental data.

### ⚠️ Carbon Risk Assessment

Classify predicted emissions into business-friendly risk categories.

### 🌱 Sustainability Recommendations

Generate actionable recommendations to improve sustainability performance and reduce emissions.

### 📉 Interactive Dashboard

Enterprise-style Streamlit dashboard with operational KPIs, analytics, forecasting, and decision-support insights.

---

## 🎯 Problem Statement

Industrial activities are among the largest contributors to global carbon emissions.

Organizations often struggle to answer important questions such as:

* Which operational factors contribute most to emissions?
* How can future emissions be predicted?
* How do emissions compare with global trends?
* What sustainability actions can reduce environmental impact?
* How can carbon-related risks be identified early?

Carbon AI addresses these challenges by transforming industrial and environmental data into measurable carbon intelligence.

---

## 🏗️ Project Architecture

```text
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Machine Learning Model Training
        ↓
Carbon Emission Prediction
        ↓
Risk Assessment
        ↓
Recommendation Engine
        ↓
Country Forecasting
        ↓
Interactive Dashboard
```

---

## 📂 Datasets Used

### Dataset 1 — Industrial Carbon Emission Dataset

Contains industrial operational and sustainability-related factors:

* Sector
* Total Energy Consumption
* Renewable Energy Consumption
* Non-Renewable Energy Consumption
* Production Output
* Supply Chain Transport Distance
* Transport Mode
* Raw Material Usage
* Process Efficiency
* Carbon Reduction Strategy
* Sustainability Metrics

### Target Variable

```text
Carbon_Emission_tCO2e_TARGET
```

### Used For

* Data Analysis
* Feature Engineering
* Machine Learning Prediction
* Carbon Risk Assessment
* Sustainability Recommendations

---

### Dataset 2 — Country CO₂ Emission Dataset

Features:

```text
country_code
country_name
year
value
```

### Used For

* Global Analytics
* Country Benchmarking
* Historical Trend Analysis
* Emission Forecasting

---

## 🧠 Machine Learning Models

### Industrial Carbon Emission Prediction

**Model:** Random Forest Regressor

#### Why Random Forest?

Random Forest was selected because it:

* Handles nonlinear relationships effectively
* Captures complex interactions between variables
* Reduces overfitting through ensemble learning
* Performs strongly on structured tabular datasets
* Supports feature importance analysis

#### Input Features

* Sector
* Energy Consumption
* Renewable Energy Usage
* Production Output
* Transport Distance
* Transport Mode
* Raw Material Usage
* Process Efficiency
* Sustainability Indicators

#### Output

```text
Predicted Carbon Emission (tCO₂e)
```

---

### Country Emission Forecasting

**Model:** Linear Regression

#### Purpose

* Forecast future CO₂ emissions
* Analyze long-term environmental trends
* Support sustainability planning

#### Forecast Horizon

```text
2026 - 2035
```
---

## 📁 Project Structure

```text
CarbonAI/
│
├── app.py
├── requirements.txt
├── models.zip
├── .gitignore
│
├── data/
│   ├── carbon_emission_dataset_with_Industry.csv
│   └── co2_emissions_kt_by_country.csv
│
├── notebooks/
│   ├── carbon_model.ipynb
│   └── country_analytics.ipynb
```

---

## ⚠️ Important Setup Instructions

The trained machine learning models are compressed to reduce repository size.

The repository contains:

```text
models.zip
```

### Step 1

Extract:

```text
models.zip
```

### Step 2

After extraction, ensure the following structure exists:

```text
models/
│
├── carbon_model.joblib
└── country_forecast.joblib
```

### Step 3

Place the extracted folder in the project root directory:

```text
CarbonAI/
│
├── app.py
├── models/
│   ├── carbon_model.joblib
│   └── country_forecast.joblib
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-url>
cd CarbonAI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

After extracting the models:

```bash
streamlit run app.py
```

Application will run locally at:

```text
http://localhost:8501
```

---

## 📈 Business Questions Solved

### Industrial Analytics

* Which sectors generate the highest emissions?
* How does energy consumption affect emissions?
* What operational factors contribute most to carbon emissions?
* How does process efficiency impact sustainability?

### Predictive Analytics

* What are the expected future emissions?
* What is the associated carbon risk?

### Global Analytics

* Which countries produce the highest CO₂ emissions?
* How have emissions evolved over time?
* How does a country compare with global trends?

### Sustainability Analytics

* What actions can reduce emissions?
* Which sustainability improvements provide the greatest impact?

---

## 🛠️ Technology Stack

### Data Analysis

* Python
* Pandas
* NumPy

### Visualization

* Plotly
* Matplotlib
* Streamlit

### Machine Learning

* Scikit-Learn
* Random Forest Regressor
* Linear Regression

### Deployment

* Streamlit

### Version Control

* Git
* GitHub

---

## 🎓 Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning
* Regression Modeling
* Forecasting
* Data Visualization
* Dashboard Development
* Sustainability Analytics
* Business Problem Solving
* Git & GitHub
* Streamlit Deployment

---

## 🔮 Future Enhancements

* SHAP Explainability
* Carbon Reduction Simulator
* Industry Benchmarking
* Real-Time Carbon Monitoring
* Advanced Forecasting Models
* Sustainability Optimization Engine

---

## 👩‍💻 Author

**Umi Salma**

Computer Science & Engineering (Data Science)

Focused on Data Science, Machine Learning, Sustainability Analytics, and building impactful data-driven solutions.

---

## 📜 License

This project is intended for educational, research, and portfolio purposes.
