# ⚾ MLB Pitcher Performance Prediction

## Objective
Our goal is to **predict a pitcher’s next season WAR (Wins Above Replacement)** using historical statistics.

## Table of Contents
1. [Explanation of WAR](#-explanation-of-war)  
2. [Data Preparation / Cleaning](#-data-preparation--cleaning)  
3. [Model Comparison](#-model-comparison)  
4. [Improving the Model](#-improving-the-model)  

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

Once the base dataset was constructed, we needed to address **missing values** and **irrelevant columns** before moving on to feature engineering.

### 5) Inspect missing values
We counted nulls across all columns:

```python
null_count = pitching.isnull().sum()
null_count
```

<img width="193" height="224" alt="image" src="https://github.com/user-attachments/assets/06ed5ebb-01fb-45b9-b06a-a66cd973b082" />

This shows some advanced metrics (e.g., Pitching+, Stf+ F0) had thousands of missing values, and Next_WAR was missing in the final season for each pitcher (as expected).

### 6) Keep only complete columns
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
- Next_WAR: our target, can’t be used as input.
- Season: not predictive on its own (would cause leakage in time splits).
- Name, Team: identifiers, not stable predictors of skill.
- IDfg: player ID, excluded if present.
After exclusions, we keep only numeric columns, since most ML regressors require numeric inputs.

### 2) Per-model preprocessing

Different models have different preprocessing needs, so I set up two pipelines:

```python
scaler = ColumnTransformer([('num', StandardScaler(), num_features)], remainder='drop')
passthrough = ColumnTransformer([('num', 'passthrough', num_features)], remainder='drop')
```
- scaler: applies standardization → used for Ridge, Lasso, and KNN.
- passthrough: leaves values as-is → used for tree ensembles like Random Forest, Gradient Boosting, and XGBoost.

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

### 3) Walk-forward back test

Before comparing models, I set up an evaluation function and a realistic validation scheme.

```python
def evaluate(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'R2': r2}
```

Then, to mimic real forecasting, I used a walk-forward scheme:
1. Train on all seasons up to year t
2. Test on year t+1
3. Repeat for every possible cutoff
4. Average results across all test years
   
This prevents data leakage and mirrors how teams would actually predict an upcoming season.

```python
seasons = sorted(df['Season'].unique())
results = []
pred_store = {}  # optional: keep per-year predictions

for cutoff in seasons[:-1]:
    train_idx = df['Season'] <= cutoff
    test_idx  = df['Season'] == (cutoff + 1)
    if not test_idx.any():
        continue

    X_train, X_test = X_all.loc[train_idx], X_all.loc[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    for name, (prep, est) in MODELS.items():
        pipe = Pipeline([('prep', prep), ('model', est)])
        pipe.fit(X_train, y_train)
        yhat = pipe.predict(X_test)

        m = evaluate(y_test, yhat)
        m.update({'Model': name, 'train_upto': int(cutoff), 'test_year': int(cutoff + 1)})
        results.append(m)

        pred_store[(name, int(cutoff + 1))] = yhat
```

### 4) Model Selection Result

After running the walk-forward backtest, I aggregated performance across all test years to see which models were most consistent.

```python
res_df = pd.DataFrame(results)
if res_df.empty:
    raise RuntimeError("No backtest splits were evaluated. Check Season coverage.")

metrics = ['RMSE', 'R2']

agg = (res_df
       .groupby('Model')[metrics]
       .agg(['mean','std'])
       .sort_values(('RMSE','mean')))

def fmt_mean_std(s):
    return f"{s['mean']:.3f} ± {s['std']:.3f}"

leaderboard = pd.DataFrame({
    'RMSE': agg['RMSE'].apply(fmt_mean_std, axis=1),
    'R2':   agg['R2'].apply(fmt_mean_std, axis=1),
})
```

With this code, we are able to see the result:

<img width="900" height="732" alt="image" src="https://github.com/user-attachments/assets/5fc20b7e-5cfb-4f30-a544-4f4e8e4cc703" />

- Lasso Regression surprisingly came out on top in terms of RMSE, showing that a simple linear model with L1 regularization can perform competitively.
- XGBoost was solid but not dominant here — possibly due to limited sample size or overfitting risk in a time-series setting.
- KNN struggled in high-dimensional space, and Gradient Boosting underperformed relative to Random Forest.
- Overall, results suggest that simpler models (Lasso, Ridge) are strong baselines, while tree ensembles are competitive but don’t always guarantee superior results in this dataset.
  
Since, Lasso Regression has the lowest RMSE and highest R², we select Lasso Regression from the 6 models we compared.

## Improving the Model

We first want to have a look on the comparison of our model's prediction to the actual WAR over time.

<img width="359" height="98" alt="image" src="https://github.com/user-attachments/assets/1ca20153-ca5e-444d-ac07-ce3ed1c20ff9" />

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/26711a03-a246-45da-8ca8-c9f46284ea02" />

The model's out-of-sample predictions for mean next-season WAR track the actual results pretty well year-over-year. The little dip around the 2019 fold makes sense—that was the year of the home run spike, which dropped pitcher WAR. My model, trained on the years before that, naturally overpredicted a bit. The 2020 fold is a completely different problem; the 60-game season was such an outlier in terms of innings and player usage that the stats just aren't comparable, which is why the model's calibration breaks down there.

So, I decided to just drop the 2020 data entirely (including the 2019→2020 backtest fold).

<img width="434" height="98" alt="image" src="https://github.com/user-attachments/assets/9bac1f95-2e98-4b63-aeab-9419bef3c8d4" />

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/6813d48d-23d5-459f-8cf2-c72f8542cef4" />

Once I removed it, the lines tracked much more closely, and the overall walk-forward fit got better—the RMSE went down and the R² went up.

To further improve the model, I decided to address highly collinear features—predictors that are strongly correlated with each other. I set a correlation threshold of 0.9 and removed any feature that exceeded it.

For many linear models, dropping highly correlated features can improve performance and make the coefficients more stable. 

<img width="312" height="475" alt="image" src="https://github.com/user-attachments/assets/39d0ac0f-6652-4323-9ab3-b49bb57ec6d9" />

However, after I re-ran my model, the results showed almost no difference in the final RMSE or R². Here, RMSE of 1.084 means that, on average, the model's prediction for a player's next-season WAR is typically off by about 1.084 wins. Also, R² of 0.383 means that the model's features explain about 38.3% of the total year-to-year variance in actual player WAR.

<img width="682" height="416" alt="image" src="https://github.com/user-attachments/assets/04a0c9bc-42bf-4f85-93ad-8ae99faa1131" />

This made sense. My model uses Lasso regression, which is already very good at handling multicollinearity on its own. By design, Lasso's regularization will select one feature from a highly correlated group and shrink the coefficients of the others to zero anyway. So, it turns out that manually removing those features beforehand didn't really add any value.

<img width="989" height="970" alt="image" src="https://github.com/user-attachments/assets/020627e1-7131-4605-97de-2e9c257c79e5" />

Now I wanted to look at feature importance, which I can read off the plot of standardized Lasso coefficients. 

With all features on the same scale, we can directly compare the bar lengths to see what's driving the predictions.

### Key Positive Signals

Last-Year WAR: By far the strongest positive signal, showing that performance persists year-over-year.

Strikeouts (SO): The next most important positive predictor.

HR/FB Rate: The small positive coefficient looks odd at first, but it makes sense when you consider regression to the mean—an elevated (unlucky) HR/FB rate often "snaps back" with better performance the following season.

### Key Negative Signals

Games Pitched (G): Swings negative, likely acting as a "reliever proxy," where a high number of appearances with fewer innings correlates with a lower next-year WAR once other factors are held fixed.

Other Logical Factors: The signs for other features line up with baseball sense: xFIP and BB/9 tilt negative, Zone% is positive, and Age is a touch negative.

### Lasso's Feature Selection

Ignored Features: Most other coefficients "hug zero," which is exactly what Lasso is supposed to do: it keeps the few features that matter and quietly shrinks the rest.

---

<img width="781" height="254" alt="image" src="https://github.com/user-attachments/assets/f657c99c-227a-485c-9e7e-2bb2d5091986" />

Finally, we can look at the top 10 pitchers for 2025 as predicted by the model and compare them to the Steamer and ZiPS projections to see how well our model holds up against industry standards. Note that at the time of this report is written, 2025 season has not ended yet so we would not be able to compare with the actual statistic. 

- Steamer is a widely respected projection system that relies heavily on a player's recent stats, using regression and weighting to account for league-wide changes and aging curves.

- ZiPS (the SZymborski Projection System) is another top system that uses multi-year stats and historical player comparisons to generate a range of potential outcomes for a player.

In summary, the model demonstrated solid predictive performance with an RMSE of 1.084, indicating its forecasts for next-season WAR are typically off by about 1.08 wins. Furthermore, with an R-squared of 0.383, the model successfully explains 38.3% of the year-to-year variance in pitcher performance.

When benchmarked against established industry projections like Steamer and ZiPS, the model proved to be quite credible, identifying a similar cohort of top-tier pitchers and producing comparable results. While more validation would be needed to match their established credibility, this strong initial alignment serves as an excellent proof-of-concept.

Looking ahead, several avenues could further enhance the model's accuracy. Incorporating and scoring historical injury data could help account for health risks, while creating a more robust distinction between starting pitchers and relievers could capture the unique performance drivers of each role. For this particular analysis, a unified approach was chosen to maintain a larger and richer dataset, but role-specific modeling is a clear next step for future iterations.

📂 Notebook: [Predictions Analysis](https://github.com/RyanKim04/Predicting-WAR/blob/main/files/predictions_analysis.ipynb)

