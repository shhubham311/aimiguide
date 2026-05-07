# Time Series Analysis

## 1. Why This Topic Matters

Time series data is everywhere: stock prices, weather patterns, website traffic, sensor readings, patient vitals, and economic indicators. Unlike cross-sectional data, time series have **temporal dependencies** — future values depend on past values. Ignoring this structure leads to data leakage and poor forecasts. Time series analysis is critical for forecasting, anomaly detection, and understanding causal dynamics in finance, healthcare, supply chains, and IoT systems.

---

## 2. Components of a Time Series

Every time series can be decomposed into components:

```
Additive Model:     Y(t) = T(t) + S(t) + R(t)
Multiplicative:     Y(t) = T(t) × S(t) × R(t)

where:
  T(t) = Trend        (long-term direction)
  S(t) = Seasonality  (repeating patterns, e.g., weekly, yearly)
  R(t) = Residual     (irregular / noise)
```

```
Time Series Decomposition:
      Y(t) = T(t) + S(t) + R(t)

Value  ─────────────────────────────────────
  ↑        ╱ T(t): Trend (upward)
  │       ╱  ╱╲  ╱╲
  │      ╱  ╱  ╲╱  ╲   S(t): Seasonal pattern
  │     ╱  ╱         ╲
  │    ╱  ╱           ╲
  │   ╱  ╱      +R(t): Random noise
  └──────────────────────────── Time
```

---

## 3. Stationarity

**Stationarity** is the fundamental assumption of most time series models. A time series is **strictly stationary** if its statistical properties (mean, variance, autocorrelation) do not change over time.

### 3.1 Types of Stationarity

| Property | Stationary | Non-Stationary |
|----------|-----------|----------------|
| Mean | Constant over time | Drifts or trends |
| Variance | Constant (homoscedastic) | Changes (volatility clustering) |
| Autocorrelation | Depends only on lag | Changes with time |

```
Stationary:                    Non-Stationary:
  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~         ╱
  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~       ╱
  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~     ╱
  ──────────────────────     ╱  (upward trend)

  Mean is constant             Mean is increasing
  Variance is stable           Variance may grow
```

### 3.2 Augmented Dickey-Fuller (ADF) Test

Tests the null hypothesis that the series has a unit root (is non-stationary).

```
H0: series is non-stationary (has unit root)
H1: series is stationary

ADF test statistic < critical value → Reject H0 → Stationary
p-value < 0.05 → Reject H0 → Stationary
```

### 3.3 Achieving Stationarity: Differencing

**First-order differencing** removes linear trends:

```
Y'(t) = Y(t) - Y(t-1)

Second-order differencing removes quadratic trends:
Y''(t) = Y'(t) - Y'(t-1) = Y(t) - 2·Y(t-1) + Y(t-2)
```

**Log transformation** stabilizes variance:

```
Y_log(t) = log(Y(t))     (for positive values only)
Y_log_diff(t) = log(Y(t)) - log(Y(t-1))
```

---

## 4. Autocorrelation and Partial Autocorrelation

### 4.1 Autocorrelation Function (ACF)

Measures the correlation between a time series and its lagged values:

```
ACF(k) = Corr(Y(t), Y(t-k))

where k = lag (number of time steps)
```

```
ACF Plot (for AR(1) with φ=0.8):

ACF   |
 1.0 ┤█
     │█
 0.8 ┤█
     │ █
 0.6 ┤ █
     │ █
 0.4 ┤  █
     │  █
 0.2 ┤   █
     │    █
 0.0 ┼─────█───█───█───█───
     0  1  2  3  4  5   Lag k

    Decays gradually → suggests AR process
```

### 4.2 Partial Autocorrelation Function (PACF)

Measures the direct correlation between Y(t) and Y(t-k), removing the effect of intermediate lags:

```
PACF(k) = Corr(Y(t), Y(t-k) | Y(t-1), Y(t-2), ..., Y(t-k+1))
```

### 4.3 Reading ACF/PACF Plots for Model Selection

| Pattern | ACF | PACF | Model |
|---------|-----|------|-------|
| AR(p) | Decays gradually | Cuts off after lag p | Autoregressive |
| MA(q) | Cuts off after lag q | Decays gradually | Moving Average |
| ARMA(p,q) | Decays gradually | Decays gradually | Mixed |

---

## 5. ARIMA Models

### 5.1 The ARIMA(p, d, q) Framework

```
ARIMA(p, d, q)

p = order of Autoregressive (AR) terms
d = degree of differencing (I, Integration)
q = order of Moving Average (MA) terms
```

**AR(p) — Autoregressive:** Current value depends on past values

```
Y(t) = c + φ₁·Y(t-1) + φ₂·Y(t-2) + ... + φ_p·Y(t-p) + ε(t)

where:
  φ_i = autoregressive coefficients
  ε(t) = white noise (iid, mean=0, constant variance)
  c    = constant (drift)
```

**MA(q) — Moving Average:** Current value depends on past forecast errors

```
Y(t) = c + ε(t) + θ₁·ε(t-1) + θ₂·ε(t-2) + ... + θ_q·ε(t-q)

where:
  θ_i = moving average coefficients
  ε(t) = white noise
```

**I(d) — Integration:** Differencing d times to achieve stationarity

```
After d differencing operations, apply ARMA(p, q) to the differenced series.
```

### 5.2 ARIMA Summary Equation

```
After differencing d times to get Y'(t):

Y'(t) = c + Σ φ_i·Y'(t-i) + Σ θ_j·ε(t-j) + ε(t)

AR part (i=1..p)        MA part (j=1..q)      Noise
```

### 5.3 SARIMA (Seasonal ARIMA)

Extends ARIMA to handle seasonality:

```
SARIMA(p, d, q)(P, D, Q, s)

(p, d, q) = non-seasonal orders
(P, D, Q) = seasonal orders
s         = seasonal period (e.g., 12 for monthly, 7 for weekly)

Example: SARIMA(1,1,1)(1,1,1,12) — common for monthly data with yearly seasonality
```

---

## 6. Forecast Evaluation Metrics

```
MAE  (Mean Absolute Error):      (1/n) Σ |Y(t) - Ŷ(t)|
MSE  (Mean Squared Error):       (1/n) Σ (Y(t) - Ŷ(t))²
RMSE (Root MSE):                 √MSE
MAPE (Mean Abs % Error):         (100/n) Σ |Y(t) - Ŷ(t)| / |Y(t)|

MAE:   Robust to outliers, easy to interpret (same units as data)
RMSE:  Penalizes large errors more (squared loss)
MAPE:  Scale-independent, useful for comparing across series
```

---

## 7. Python Implementation

### 7.1 Exploratory Time Series Analysis

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Load airline passengers dataset (classic time series)
# Monthly airline passengers (1949-1960)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
df.columns = ['Passengers']

print(f"Dataset: {len(df)} months, from {df.index.min()} to {df.index.max()}")
print(df.head())

# Basic statistics
print(f"\nMean passengers: {df['Passengers'].mean():.1f}")
print(f"Std passengers:  {df['Passengers'].std():.1f}")
print(f"Min: {df['Passengers'].min()}, Max: {df['Passengers'].max()}")
```

### 7.2 Stationarity Testing and Differencing

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, title=''):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value:       {result[1]:.6f}')
    print(f'Critical Values:')
    for key, val in result[4].items():
        print(f'  {key}: {val:.4f}')
    if result[1] < 0.05:
        print('→ Stationary (reject H0)')
    else:
        print('→ Non-stationary (fail to reject H0)')

# Test original series
print("=== Original Series ===")
adf_test(df['Passengers'])

# First difference
df['Diff1'] = df['Passengers'].diff()
print("\n=== First Difference ===")
adf_test(df['Diff1'])

# Log-transformed first difference
df['LogDiff'] = np.log(df['Passengers']).diff()
print("\n=== Log First Difference ===")
adf_test(df['LogDiff'])
# Log first difference typically becomes stationary
```

### 7.3 ACF and PACF Analysis

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df['Passengers'].dropna(), lags=40, ax=axes[0])
plot_pacf(df['Passengers'].dropna(), lags=40, ax=axes[1])
axes[0].set_title('ACF - Original Series')
axes[1].set_title('PACF - Original Series')
plt.tight_layout()
plt.savefig('acf_pacf_original.png', dpi=100)

# ACF and PACF of differenced series
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df['LogDiff'].dropna(), lags=40, ax=axes[0])
plot_pacf(df['LogDiff'].dropna(), lags=40, ax=axes[1])
axes[0].set_title('ACF - Log First Difference')
axes[1].set_title('PACF - Log First Difference')
plt.tight_layout()
plt.savefig('acf_pacf_diff.png', dpi=100)

# Interpretation:
# ACF of diff series: significant at lag 1, and at seasonal lag 12 → suggests MA(1) + seasonal
# PACF of diff series: significant at lag 1 → suggests AR(1)
# → Candidate model: SARIMA(1,1,1)(1,1,1,12)
```

### 7.4 Fitting ARIMA and SARIMA

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Train/test split (last 24 months for testing)
train = df['Passengers'][:-24]
test = df['Passengers'][-24:]

# ---- ARIMA(2,1,2) ----
arima = ARIMA(train, order=(2, 1, 2))
arima_fit = arima.fit()
arima_forecast = arima_fit.forecast(steps=24)

arima_mae = mean_absolute_error(test, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
print(f"ARIMA(2,1,2)  MAE: {arima_mae:.2f}  RMSE: {arima_rmse:.2f}")

# ---- SARIMA(1,1,1)(1,1,1,12) ----
sarima = SARIMAX(train,
                 order=(1, 1, 1),
                 seasonal_order=(1, 1, 1, 12),
                 enforce_stationarity=False)
sarima_fit = sarima.fit(disp=False)
sarima_forecast = sarima_fit.forecast(steps=24)

sarima_mae = mean_absolute_error(test, sarima_forecast)
sarima_rmse = np.sqrt(mean_squared_error(test, sarima_forecast))
print(f"SARIMA(1,1,1)(1,1,1,12)  MAE: {sarima_mae:.2f}  RMSE: {sarima_rmse:.2f}")

# SARIMA typically outperforms ARIMA for seasonal data

# Print summary
print(sarima_fit.summary())
```

### 7.5 Auto-ARIMA (Automated Model Selection)

```python
# Manual grid search for best ARIMA order
import warnings
warnings.filterwarnings('ignore')

best_aic = float('inf')
best_order = None

for p in range(0, 4):
    for d in range(0, 3):
        for q in range(0, 4):
            try:
                model = ARIMA(train, order=(p, d, q))
                fit = model.fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p, d, q)
            except:
                continue

print(f"Best ARIMA order: {best_order}  (AIC: {best_aic:.2f})")
# Typically: (2,1,2) or (3,1,1) for airline data

# Fit and evaluate best model
best_model = ARIMA(train, order=best_order).fit()
best_forecast = best_model.forecast(steps=24)
best_mae = mean_absolute_error(test, best_forecast)
print(f"Best model MAE: {best_mae:.2f}")
```

### 7.6 Rolling Forecast (Walk-Forward Validation)

```python
# More realistic evaluation: predict one step ahead, then roll forward
history = [x for x in train]
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=(2, 1, 2))
    model_fit = model.fit()
    yhat = model_fit.forecast(steps=1)[0]
    predictions.append(yhat)
    history.append(test.iloc[t])

rolling_mae = mean_absolute_error(test, predictions)
rolling_rmse = np.sqrt(mean_squared_error(test, predictions))
print(f"\nRolling Forecast (1-step ahead):")
print(f"  MAE:  {rolling_mae:.2f}")
print(f"  RMSE: {rolling_rmse:.2f}")
# Rolling forecasts are more realistic than static multi-step forecasts
```

---

## 8. Exponential Smoothing

An alternative to ARIMA that is computationally simpler:

```
Simple Exponential Smoothing:
  F(t+1) = α·Y(t) + (1-α)·F(t)

  α ∈ (0, 1): smoothing parameter
  α close to 1: more weight on recent observations
  α close to 0: more weight on past forecast (smoother)

Double Exponential Smoothing (Holt's method): adds trend component
Triple Exponential Smoothing (Holt-Winters): adds seasonal component
```

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Holt-Winters (triple exponential smoothing)
hw = ExponentialSmoothing(
    train,
    seasonal_periods=12,
    trend='mul',
    seasonal='mul',
    damped_trend=True
)
hw_fit = hw.fit()
hw_forecast = hw_fit.forecast(24)

hw_mae = mean_absolute_error(test, hw_forecast)
hw_rmse = np.sqrt(mean_squared_error(test, hw_forecast))
print(f"Holt-Winters MAE: {hw_mae:.2f}  RMSE: {hw_rmse:.2f}")
# Competitive with SARIMA, but faster to fit
```

---

## 9. Real-World ML Connections

- **Finance:** ARIMA/GARCH models for stock price prediction, volatility forecasting, and risk management (Value at Risk).
- **Supply Chain:** SARIMA forecasts demand for inventory optimization at Amazon, Walmart, and other retailers.
- **Energy:** Time series models predict electricity demand and renewable energy output (solar, wind) for grid management.
- **Healthcare:** Patient vital sign monitoring uses anomaly detection on time series to trigger early warnings in ICUs.
- **Web Analytics:** Forecasting page views, server load, and user engagement using seasonal decomposition and exponential smoothing.
- **Machine Learning Integration:** Time series features (lags, rolling stats, Fourier terms) are commonly fed into XGBoost and neural networks for hybrid models.

---

## 10. Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Ignoring stationarity | Always test with ADF test; difference if needed |
| Data leakage in train/test | Split by time, never randomly |
| Ignoring seasonality | Use SARIMA or Holt-Winters |
| Overfitting ARIMA | Use AIC/BIC for model selection |
| Not checking residuals | Residuals should be white noise (check ACF of residuals) |
| Confusing forecast types | One-step vs multi-step: rolling validation for realistic evaluation |

---

## 11. Exercises

### Exercise 1: Seasonal Decomposition
Decompose the airline passengers dataset using both additive and multiplicative models. Which is more appropriate? Justify by comparing the residual variance.

### Exercise 2: ARIMA Grid Search
Perform a grid search over ARIMA(p,d,q) with p∈[0,3], d∈[0,2], q∈[0,3] on the airline dataset. Select the model with the lowest AIC. Compare its test MAE with SARIMA.

### Exercise 3: Residual Diagnostics
Fit a SARIMA model and check if residuals are white noise by: (a) plotting ACF of residuals, (b) running Ljung-Box test.

### Exercise 4: Holt-Winters Comparison
Compare Simple ES, Holt's (double), and Holt-Winters (triple) on the airline dataset. Report test MAE for each.

### Solutions

**Solution 1:**
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Additive decomposition
add_decomp = seasonal_decompose(df['Passengers'], model='additive', period=12)
add_resid_var = np.var(add_decomp.resid.dropna())

# Multiplicative decomposition
mul_decomp = seasonal_decompose(df['Passengers'], model='multiplicative', period=12)
mul_resid_var = np.var(mul_decomp.resid.dropna())

print(f"Additive residual variance:    {add_resid_var:.4f}")
print(f"Multiplicative residual var:   {mul_resid_var:.4f}")
print(f"Multiplicative is better: {mul_resid_var < add_resid_var}")
# Multiplicative is better because the seasonal pattern grows with the trend
```

**Solution 2:**
```python
import itertools
best_aic, best_params = float('inf'), None

for p, d, q in itertools.product(range(4), range(3), range(4)):
    try:
        model = ARIMA(train, order=(p, d, q))
        fit = model.fit()
        if fit.aic < best_aic:
            best_aic, best_params = fit.aic, (p, d, q)
    except:
        continue

print(f"Best ARIMA{best_params}, AIC={best_aic:.2f}")

model = ARIMA(train, order=best_params).fit()
pred = model.forecast(24)
print(f"Test MAE: {mean_absolute_error(test, pred):.2f}")
```

**Solution 3:**
```python
from statsmodels.stats.diagnostic import acorr_ljungbox

model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
residuals = model.resid

# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print("Ljung-Box Test:")
print(lb_test)
# p-value > 0.05 → residuals are white noise (good model)

# ACF of residuals
fig, ax = plt.subplots(figsize=(10, 3))
plot_acf(residuals, lags=30, ax=ax)
ax.set_title('ACF of Residuals')
plt.tight_layout()
plt.savefig('residual_acf.png')
# Residual ACF should show no significant autocorrelation
```

**Solution 4:**
```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Simple Exponential Smoothing
ses = SimpleExpSmoothing(train).fit()
ses_pred = ses.forecast(24)
print(f"SES MAE: {mean_absolute_error(test, ses_pred):.2f}")

# Double (Holt's)
holt = ExponentialSmoothing(train, trend='add', damped_trend=True).fit()
holt_pred = holt.forecast(24)
print(f"Holt's MAE: {mean_absolute_error(test, holt_pred):.2f}")

# Triple (Holt-Winters)
hw = ExponentialSmoothing(train, trend='mul', seasonal='mul',
                          seasonal_periods=12, damped_trend=True).fit()
hw_pred = hw.forecast(24)
print(f"Holt-Winters MAE: {mean_absolute_error(test, hw_pred):.2f}")
# Holt-Winters typically wins for seasonal data
```

---

## 12. Summary

| Concept | Key Takeaway |
|---------|-------------|
| Stationarity | Core assumption; test with ADF; achieve with differencing |
| ACF / PACF | Essential for identifying AR and MA orders |
| ARIMA(p,d,q) | AR + differencing + MA; handles non-seasonal data |
| SARIMA | Extends ARIMA with seasonal components |
| AIC/BIC | Information criteria for model selection (lower = better) |
| Exponential Smoothing | Simpler alternative; Holt-Winters handles seasonality |
| Residual Diagnostics | Residuals must be white noise; check ACF and Ljung-Box |
| Rolling Forecast | Most realistic evaluation; predicts 1 step, rolls forward |
