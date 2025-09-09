# ⚾ MLB Pitcher Performance Prediction

## Objective
Our goal is to **predict a pitcher’s next season WAR (Wins Above Replacement)** using historical statistics.

## Table of Contents
1. [Explanation of WAR](#-explanation-of-war)  
2. [Data Preparation / Cleaning](#-data-preparation--cleaning)  
3. [Model Comparison](#-model-comparison)  
4. [Further Analysis](#-further-analysis)  
5. [Results & Takeaways](#-results--takeaways)

## Explanation of WAR

**Wins Above Replacement (WAR)** is a comprehensive baseball statistic designed to summarize a player’s total contributions to their team into a single number.  
It answers the question:

*“How many more wins does this player contribute compared to a replacement-level player (a readily available minor-leaguer or bench player)?”*

### Why WAR Matters
- WAR attempts to capture **both quality and quantity** of a player’s contributions.  
- For pitchers, WAR accounts for:
  - Innings pitched
  - Runs allowed (adjusted for defense & park factors)
  - Strikeouts, walks, and home runs  
- It allows fair comparisons across **seasons, teams, and even eras**.

### Interpreting WAR for Pitchers
The following ranges give a sense of how WAR translates to on-field ability:

| WAR   | Ability                          |
|-------|----------------------------------|
| 0–1   | Replacement level                |
| 1–2   | Utility player                   |
| 2–3   | Average starter                  |
| 3–4   | Above average starter            |
| 4–5   | All-Star level player            |
| 5–6   | Superstar, very well-known player|
| 6+    | MVP level                        |

For example,
- A pitcher with **WAR = 3.5** is considered an **above average starter**.  
- A pitcher with **WAR = 6.2** is performing at an **MVP level**, contributing 6+ wins more than a replacement-level pitcher.


### Why We Predict WAR
In this project, our target is to predict **next season’s pitcher WAR**:
- WAR captures both **durability** (innings pitched) and **quality** (ERA, FIP, etc.).
- It’s a single, widely understood number → perfect for benchmarking model accuracy.
- WAR is valuable to **teams, analysts, and fans** for forecasting player impact.


## Data Preparation / Cleaning
- **Source**: [pybaseball](https://github.com/jldbc/pybaseball) for MLB stats  
- **Steps**:
  - Handle missing values (null handling)  
  - Drop highly collinear features  

📂 Notebook: [Data Cleaning](notebooks/01-data-cleaning.ipynb)

## Model Comparison
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

## Further Analysis
- **2024 Prediction**: Compare predicted WAR vs. actual WAR (when available).  
- **Limitations**:
  - Feature availability (injury history, pitch types, etc.)  
  - Small-sample bias for rookies  
- **Areas of Improvement**:
  - Incorporate Statcast data  
  - Try deep learning approaches (RNN, Transformers)  

📂 Notebook: [Further Analysis](notebooks/03-further-analysis.ipynb)

## Results & Takeaways
- Random Forest and XGBoost performed best on validation metrics.  
- Predictions for 2024 WAR show reasonable alignment with early-season actuals.  

![Sample Plot](assets/sample-plot.png)

## 📂 Repository Structure

## 🙌 About
This project was built for exploring **sports analytics + machine learning**.  
Feel free to fork, run, or contribute!  

🔗 Connect: [LinkedIn](https://linkedin.com/in/yourname) | [GitHub](https://github.com/yourusername)
