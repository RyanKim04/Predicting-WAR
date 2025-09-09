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
### Data Preparation

The first step was to construct a dataset where the **target** is each pitcher’s **next-season WAR**.  
We pulled historical pitching stats with the [pybaseball](https://github.com/jldbc/pybaseball) library and transformed them to fit this supervised learning setup.

### 1) Collect pitching stats
We retrieved data from **2002–2024**, filtering for pitchers with at least 50 innings pitched to avoid unstable small-sample WAR values.

```python
from pybaseball import pitching_stats

START, END = 2002, 2024
pitching = pitching_stats(START, END, qual=50)
```

### 2) Filtering
We only kept pitchers with 2+ seasons so that we can have a next-season WAR label on our dataset.

```python
pitching = pitching.groupby("IDfg", group_keys=False).filter(lambda g: g.shape[0] > 1)
```

### 3) Define the prediction target
For each pitcher, we sorted their seasons chronologically and created a new column Next_WAR, which represents the WAR in the following season.

```python
def next_season(player):
    player = player.sort_values("Season")
    player["Next_WAR"] = player["WAR"].shift(-1)
    return player

pitching = pitching.groupby("IDfg", group_keys=False).apply(next_season)
```
### 4) Outcome
Now, each row in the dataset represents:
Features: performance metrics from season t
Label: Next_WAR from season t+1
This setup mirrors the real-world task of forecasting how a pitcher will perform in the upcoming year.

<img width="1157" height="394" alt="image" src="https://github.com/user-attachments/assets/9d518e2f-f622-41cb-b9de-f92520f92ec2" />

### Data Cleaning
Once the base dataset was constructed, we needed to address **missing values** and **irrelevant columns** before moving on to feature engineering.

### 1) Inspect missing values
We counted nulls across all columns:

```python
null_count = pitching.isnull().sum()
null_count
```

<img width="193" height="224" alt="image" src="https://github.com/user-attachments/assets/06ed5ebb-01fb-45b9-b06a-a66cd973b082" />

This shows some advanced metrics (e.g., Pitching+, Stf+ F0) had thousands of missing values, and Next_WAR was missing in the final season for each pitcher (as expected).

### 2) Keep only complete columns
We created a subset of columns with no missing values, and explicitly added back Next_WAR as the prediction target.

```python
complete_cols = list(pitching.columns[null_count == 0])
pitching = pitching[complete_cols + ["Next_WAR"]].copy()
```

## Model Comparison
After data cleaning, the next step was to benchmark several machine learning models for predicting **next-season WAR**.  

We will be comparing the models below and choose the best model by comparing evaluation metrics (RMSE, R²)

- **Linear models**: Ridge, Lasso (require scaling to handle feature magnitudes)
- **Distance-based**: k-Nearest Neighbors (KNN)
- **Tree ensembles**: Random Forest, Gradient Boosting, XGBoost (nonlinear, capture interactions)

### 1) Feature Set
We begin by defining the **feature set** for model training and separating the prediction target.

```python
exclude = {'Next_WAR', 'Season', 'Name', 'Team'}
if 'IDfg' in df.columns:
    exclude.add('IDfg')

feature_cols = [c for c in df.columns if c not in exclude]
X_all = df[feature_cols].select_dtypes(include=[np.number]).copy()
y_all = df['Next_WAR'].values
num_features = X_all.columns.tolist()
```
We remove columns that would cause data leakage or don’t serve as predictive features:
Next_WAR: our target, can’t be used as input.
Season: not predictive on its own (would cause leakage in time splits).
Name, Team: identifiers, not stable predictors of skill.
IDfg: player ID, excluded if present.
After exclusions, we keep only numeric columns, since most ML regressors require numeric inputs.

### 2) Per-model preprocessing

Different models have different preprocessing needs, so I set up two pipelines:

```python
scaler = ColumnTransformer([('num', StandardScaler(), num_features)], remainder='drop')
passthrough = ColumnTransformer([('num', 'passthrough', num_features)], remainder='drop')
```
scaler: applies standardization → used for Ridge, Lasso, and KNN.
passthrough: leaves values as-is → used for tree ensembles like Random Forest, Gradient Boosting, and XGBoost.

Next, I registered the models with their appropriate preprocessing:
```python
MODELS = {
    'Ridge': (scaler, Ridge(alpha=5.0, random_state=42)),
    'Lasso': (scaler, Lasso(alpha=0.01, max_iter=20000, random_state=42)),
    'KNN': (scaler, KNeighborsRegressor(n_neighbors=15, weights='distance', p=2)),

    'RandomForest': (passthrough, RandomForestRegressor(
        n_estimators=600, min_samples_leaf=2, n_jobs=-1, random_state=42
    )),
    'GradientBoosting': (passthrough, GradientBoostingRegressor(
        learning_rate=0.05, n_estimators=800, subsample=0.9, max_depth=3, random_state=42
    )),
    'XGBoost': (passthrough, XGBRegressor(
        n_estimators=1200, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )),
}
```



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
