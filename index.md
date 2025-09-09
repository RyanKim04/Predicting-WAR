# ⚾ MLB Pitcher Performance Prediction

## Objective
Our goal is to **predict a pitcher’s next season WAR (Wins Above Replacement)** using historical statistics.

## 📖 Table of Contents
1. [Explanation of WAR](#-explanation-of-war)  
2. [Data Preparation / Cleaning](#-data-preparation--cleaning)  
3. [Model Comparison](#-model-comparison)  
4. [Further Analysis](#-further-analysis)  
5. [Results & Takeaways](#-results--takeaways)


## Explanation of WAR
- WAR measures a player’s overall contribution to their team.  
- **Objective**: Predict next year’s WAR for pitchers, based on past performance.  

---

## 🛠️ Data Preparation / Cleaning
- **Source**: [pybaseball](https://github.com/jldbc/pybaseball) for MLB stats  
- **Steps**:
  - Handle missing values (null handling)  
  - Drop highly collinear features  

📂 Notebook: [Data Cleaning](notebooks/01-data-cleaning.ipynb)

---

## 🤖 Model Comparison
We performed a **walk-forward backtest** (train–test split by season) and evaluated multiple algorithms:

1. Ridge Regression  
2. Lasso Regression  
3. Random Forest  
4. XGBoost  
5. K-Nearest Neighbors (KNN)  
6. Gradient Boosting  

✅ Best model selected by comparing evaluation metrics (e.g., RMSE, R², MAE).

📂 Notebook: [Model Training](notebooks/02-model-training.ipynb)  
📄 Report: [Model Comparison (HTML)](reports/model-comparison.html)

---

## 🔍 Further Analysis
- **2024 Prediction**: Compare predicted WAR vs. actual WAR (when available).  
- **Limitations**:
  - Feature availability (injury history, pitch types, etc.)  
  - Small-sample bias for rookies  
- **Areas of Improvement**:
  - Incorporate Statcast data  
  - Try deep learning approaches (RNN, Transformers)  

📂 Notebook: [Further Analysis](notebooks/03-further-analysis.ipynb)

---

## 📊 Results & Takeaways
- Random Forest and XGBoost performed best on validation metrics.  
- Predictions for 2024 WAR show reasonable alignment with early-season actuals.  

![Sample Plot](assets/sample-plot.png)

---

## 📂 Repository Structure


---

## 🙌 About
This project was built for exploring **sports analytics + machine learning**.  
Feel free to fork, run, or contribute!  

🔗 Connect: [LinkedIn](https://linkedin.com/in/yourname) | [GitHub](https://github.com/yourusername)
