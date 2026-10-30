import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import STL

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (13, 5)
RANDOM_STATE = 42

url = 'https://raw.githubusercontent.com/plotly/datasets/master/monthly-milk-production-pounds.csv'
df = pd.read_csv(url)

df.columns = ['Date', 'MilkProduction']
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
df['MilkProduction'] = pd.to_numeric(df['MilkProduction'], errors='coerce')
df = df.dropna(subset=['MilkProduction']).copy()

print('Размер датасета:', df.shape)
print('Период:', df['Date'].min().date(), '—', df['Date'].max().date())
print('Пропуски:')
print(df.isna().sum().to_frame('Количество'))
print(df.head())

plt.figure(figsize=(15, 5))
plt.plot(df['Date'], df['MilkProduction'], marker='o', linewidth=1)
plt.title('Ежемесячное производство молока на одну корову')
plt.xlabel('Дата')
plt.ylabel('Производство молока, фунтов')
plt.grid(True)
plt.show()

def make_features(data):
    out = data[['Date', 'MilkProduction']].copy()
    out['Year'] = out['Date'].dt.year
    out['Month'] = out['Date'].dt.month
    out['Quarter'] = out['Date'].dt.quarter

    out['Month_Sin'] = np.sin(2 * np.pi * out['Month'] / 12)
    out['Month_Cos'] = np.cos(2 * np.pi * out['Month'] / 12)

    for lag in [1, 2, 3, 6, 12]:
        out[f'Lag_{lag}'] = out['MilkProduction'].shift(lag)

    past = out['MilkProduction'].shift(1)
    for window in [3, 6, 12]:
        out[f'Rolling_Mean_{window}'] = past.rolling(window).mean()
        out[f'Rolling_Median_{window}'] = past.rolling(window).median()
        out[f'Rolling_Std_{window}'] = past.rolling(window).std()
        out[f'Rolling_Min_{window}'] = past.rolling(window).min()
        out[f'Rolling_Max_{window}'] = past.rolling(window).max()

    return out

df_feat = make_features(df).dropna().reset_index(drop=True)
features = [c for c in df_feat.columns if c not in ['Date', 'MilkProduction']]

print('Строк после генерации признаков:', len(df_feat))
print('Количество признаков:', len(features))
print(df_feat.head())


train_size = int(len(df_feat) * 0.8)
train = df_feat.iloc[:train_size].copy()
test = df_feat.iloc[train_size:].copy()

X_train, y_train = train[features], train['MilkProduction']
X_test, y_test = test[features], test['MilkProduction']

print('Train:', train['Date'].min().date(), '—', train['Date'].max().date(), '| строк:', len(train))
print('Test: ', test['Date'].min().date(), '—', test['Date'].max().date(), '| строк:', len(test))

tscv = TimeSeriesSplit(n_splits=5)
cv_rmse = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
    model_cv = Ridge(alpha=10.0)
    model_cv.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    pred_cv = model_cv.predict(X_train.iloc[val_idx])
    rmse = np.sqrt(mean_squared_error(y_train.iloc[val_idx], pred_cv))
    cv_rmse.append(rmse)
    print(f'Фолд {fold}: RMSE = {rmse:.3f}')

print('Средний RMSE по фолдам:', round(np.mean(cv_rmse), 3))

model = Ridge(alpha=10.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


def make_next_features(history, date, feature_names):
    s = pd.Series(history, dtype=float)
    month = date.month
    row = {
        'Year': date.year,
        'Month': month,
        'Quarter': date.quarter,
        'Month_Sin': np.sin(2 * np.pi * month / 12),
        'Month_Cos': np.cos(2 * np.pi * month / 12)
    }

    for lag in [1, 2, 3, 6, 12]:
        row[f'Lag_{lag}'] = s.iloc[-lag]

    for window in [3, 6, 12]:
        w = s.iloc[-window:]
        row[f'Rolling_Mean_{window}'] = w.mean()
        row[f'Rolling_Median_{window}'] = w.median()
        row[f'Rolling_Std_{window}'] = w.std()
        row[f'Rolling_Min_{window}'] = w.min()
        row[f'Rolling_Max_{window}'] = w.max()

    return pd.DataFrame([row])[feature_names]

H = 12
last_train_date = train['Date'].iloc[-1]
history = df.loc[df['Date'] <= last_train_date, 'MilkProduction'].tolist()
forecast_dates = pd.date_range(last_train_date + pd.offsets.MonthBegin(1), periods=H, freq='MS')

recursive_predictions = []
for forecast_date in forecast_dates:
    x_next = make_next_features(history, forecast_date, features)
    prediction = model.predict(x_next)[0]
    recursive_predictions.append(prediction)
    history.append(prediction)

actual_horizon = df.set_index('Date')['MilkProduction'].reindex(forecast_dates)

plt.figure(figsize=(14, 5))
plt.plot(forecast_dates, actual_horizon.values, marker='o', label='Фактические значения')
plt.plot(forecast_dates, recursive_predictions, marker='x', linestyle='--', label='Рекурсивный прогноз')
plt.title(f'Рекурсивный прогноз производства молока на {H} месяцев')
plt.xlabel('Дата')
plt.ylabel('Производство молока, фунтов')
plt.legend()
plt.grid(True)
plt.show()


def calculate_metrics(y_true, y_hat):
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    mse = mean_squared_error(y_true, y_hat)
    mae = mean_absolute_error(y_true, y_hat)
    nonzero = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[nonzero] - y_hat[nonzero]) / y_true[nonzero])) * 100
    wape = np.sum(np.abs(y_true - y_hat)) / np.sum(np.abs(y_true)) * 100
    return {'MSE': mse, 'MAE': mae, 'MAPE, %': mape, 'WAPE, %': wape}

train_pred = model.predict(X_train)
metrics = pd.DataFrame([
    {'Выборка': 'Train', **calculate_metrics(y_train, train_pred)},
    {'Выборка': 'Test', **calculate_metrics(y_test, y_pred)}
])
print(metrics.round(3))

test_result = test[['Date', 'MilkProduction']].copy()
test_result['Prediction'] = y_pred
test_result['Error'] = test_result['MilkProduction'] - test_result['Prediction']
test_result['Absolute_Error'] = test_result['Error'].abs()

plt.figure(figsize=(14, 5))
plt.plot(test_result['Date'], test_result['MilkProduction'], marker='o', label='Факт')
plt.plot(test_result['Date'], test_result['Prediction'], marker='x', linestyle='--', label='Прогноз')
plt.title('Фактические и предсказанные значения на тестовой выборке')
plt.xlabel('Дата')
plt.ylabel('Производство молока, фунтов')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 4))
plt.scatter(test_result['Date'], test_result['Error'], s=25)
plt.axhline(0, color='red', linestyle='--')
plt.title('Остатки модели на тестовой выборке')
plt.xlabel('Дата')
plt.ylabel('Факт − прогноз')
plt.grid(True)
plt.show()

print('Наблюдения с наибольшей абсолютной ошибкой:')
print(test_result.nlargest(5, 'Absolute_Error')[
    ['Date', 'MilkProduction', 'Prediction', 'Error', 'Absolute_Error']
])



test_result['Month'] = test_result['Date'].dt.month
monthly_error = test_result.groupby('Month')['Absolute_Error'].mean()

plt.figure(figsize=(12, 4))
monthly_error.plot(kind='bar')
plt.title('Средняя абсолютная ошибка по календарным месяцам')
plt.xlabel('Месяц')
plt.ylabel('MAE')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

print(monthly_error.round(3).to_frame('MAE по месяцу'))


series = df.set_index('Date')['MilkProduction'].asfreq('MS')

series = series.interpolate(method='time')

stl = STL(series, period=12, robust=True)
stl_result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
stl_result.observed.plot(ax=axes[0], title='Исходный ряд')
stl_result.trend.plot(ax=axes[1], title='Тренд')
stl_result.seasonal.plot(ax=axes[2], title='Сезонная компонента')
stl_result.resid.plot(ax=axes[3], title='Остаток')
plt.tight_layout()
plt.show()



df_extra = df_feat.copy()
df_extra['Past_Trend_12'] = df_extra['MilkProduction'].shift(1).rolling(12).mean()
df_extra['Past_Seasonal_12'] = df_extra['MilkProduction'].shift(12)
df_extra = df_extra.dropna().reset_index(drop=True)

extra_features = features + ['Past_Trend_12', 'Past_Seasonal_12']
split_extra = int(len(df_extra) * 0.8)
tr_e = df_extra.iloc[:split_extra]
te_e = df_extra.iloc[split_extra:]

y_tr_e = tr_e['MilkProduction']
y_te_e = te_e['MilkProduction']

base_same_rows = Ridge(alpha=10.0).fit(tr_e[features], y_tr_e)
extra_model = Ridge(alpha=10.0).fit(tr_e[extra_features], y_tr_e)

pred_base = base_same_rows.predict(te_e[features])
pred_extra = extra_model.predict(te_e[extra_features])

comparison = pd.DataFrame([
    {'Модель': 'Базовые признаки', **calculate_metrics(y_te_e, pred_base)},
    {'Модель': 'Базовые + тренд/сезонность', **calculate_metrics(y_te_e, pred_extra)}
])
print(comparison.round(3))