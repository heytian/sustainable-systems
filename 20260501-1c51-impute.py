"""
CO2 Imputation Comparison
Methods: Mean, Interpolation, LSTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(42)

FILE_PATH  = "datasource/co2_sam.csv"
WINDOW     = 12     # LSTM lookback in months
EPOCHS     = 300
LR         = 0.001
MASK_RATE  = 0.2

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD FULL FILE IN CHUNKS → MONTHLY AGGREGATES
# ─────────────────────────────────────────────────────────────────────────────
chunks = []
for chunk in pd.read_csv(FILE_PATH,
                          chunksize=200_000,
                          usecols=['city', 'datetime', 'xco2']):
    chunk = chunk.dropna()
    chunk['datetime'] = pd.to_datetime(chunk['datetime'], errors='coerce')
    chunk = chunk[chunk['xco2'].between(350, 500)]
    chunk['month'] = chunk['datetime'].dt.to_period('M')
    chunks.append(chunk.groupby(['city', 'month'])['xco2'].mean().reset_index())

monthly = (pd.concat(chunks)
             .groupby(['city', 'month'])['xco2']
             .mean()
             .reset_index())

print(f"Cities: {monthly['city'].nunique()} | "
      f"Date range: {monthly['month'].min()} → {monthly['month'].max()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SELECT BEST CITY BY TEMPORAL COVERAGE
# ─────────────────────────────────────────────────────────────────────────────
city_stats = monthly.groupby('city').agg(
    n_months=('month', 'nunique'),
    date_min=('month', 'min'),
    date_max=('month', 'max'),
).reset_index()
city_stats['span']     = city_stats.apply(
    lambda r: (r['date_max'] - r['date_min']).n + 1, axis=1)
city_stats['coverage'] = city_stats['n_months'] / city_stats['span']

best_city = (city_stats[city_stats['span'] >= 48]
             .sort_values(['n_months', 'coverage'], ascending=False)
             .iloc[0]['city'])

stats = city_stats[city_stats['city'] == best_city].iloc[0]
print(f"\nCity     : {best_city}")
print(f"Months   : {stats['n_months']}  |  "
      f"Span: {stats['span']}  |  Coverage: {stats['coverage']:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD CONTINUOUS MONTHLY SERIES
# ─────────────────────────────────────────────────────────────────────────────
city_df  = (monthly[monthly['city'] == best_city]
            .set_index('month')
            .sort_index())
full_idx = pd.period_range(city_df.index.min(),
                           city_df.index.max(), freq='M')
city_df  = city_df.reindex(full_idx)

originally_missing = np.isnan(city_df['xco2'].values)
series = (pd.Series(city_df['xco2'].values.astype(float))
            .interpolate()
            .bfill()
            .ffill()
            .values)

print(f"Series length   : {len(series)}")
print(f"Structural gaps : {originally_missing.sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ARTIFICIAL MASKING (observed months only)
# ─────────────────────────────────────────────────────────────────────────────
observed_idx     = np.where(~originally_missing)[0]
n_mask           = max(12, int(MASK_RATE * len(observed_idx)))
masked_idx       = np.random.choice(observed_idx, size=n_mask, replace=False)
mask             = np.zeros(len(series), dtype=bool)
mask[masked_idx] = True
x_missing        = series.copy()
x_missing[mask]  = np.nan

print(f"Masked points   : {mask.sum()}  ({mask.sum()/len(series):.1%})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BASELINE IMPUTERS
# ─────────────────────────────────────────────────────────────────────────────
# mean_pred   = np.where(mask, np.nanmean(x_missing), series)

knn_raw     = KNNImputer(n_neighbors=5).fit_transform(
                  x_missing.reshape(-1, 1)).flatten()
knn_pred    = np.where(mask, knn_raw, series)

interp_pred = pd.Series(x_missing).interpolate().bfill().ffill().values


# ─────────────────────────────────────────────────────────────────────────────
# 6. LSTM IMPUTER
# ─────────────────────────────────────────────────────────────────────────────
scaler       = StandardScaler()
series_norm  = scaler.fit_transform(series.reshape(-1, 1)).flatten()
missing_norm = series_norm.copy()
missing_norm[mask] = np.nan
filled_norm  = np.nan_to_num(missing_norm, nan=0.0)

def make_sequences(values, window):
    X, y = [], []
    for i in range(len(values) - window):
        target = values[i + window]
        if np.isnan(target):
            continue
        X.append(np.nan_to_num(values[i:i + window], nan=0.0))
        y.append(target)
    return (torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(y), dtype=torch.float32))

X_train, y_train = make_sequences(filled_norm, WINDOW)

class LSTMModel(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

model     = LSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    loss = criterion(model(X_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 75 == 0:
        print(f"  epoch {epoch:3d}  loss {loss.item():.5f}")

lstm_norm = missing_norm.copy()
model.eval()
with torch.no_grad():
    for i in range(len(series) - WINDOW):
        t = i + WINDOW
        if mask[t]:
            win = np.nan_to_num(missing_norm[i:i + WINDOW], nan=0.0)
            x_t = torch.tensor(
                win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            lstm_norm[t] = model(x_t).item()

lstm_pred = scaler.inverse_transform(
                lstm_norm.reshape(-1, 1)).flatten()

# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def rmse(pred, true, m):
    return np.sqrt(np.nanmean((pred[m] - true[m]) ** 2))

def bias(pred, true, m):
    return np.nanmean(pred[m] - true[m])

def get_season(month):
    return {12:'Winter',1:'Winter', 2:'Winter',
             3:'Spring', 4:'Spring', 5:'Spring',
             6:'Summer', 7:'Summer', 8:'Summer',
             9:'Autumn',10:'Autumn',11:'Autumn'}[month]

seasons_arr = np.array([get_season(m) for m in full_idx.month])
season_list = ['Winter', 'Spring', 'Summer', 'Autumn']

methods = {
    'KNN':          knn_pred,
    'Interpolation': interp_pred,
    'LSTM':          lstm_pred,
}

# Overall
print(f"\n{'Method':<16} {'RMSE':>7} {'Bias':>7}")
print("─" * 32)
for name, pred in methods.items():
    print(f"{name:<16} {rmse(pred, series, mask):7.4f} "
          f"{bias(pred, series, mask):7.4f}")

# Seasonal
print(f"\n{'Method':<16} " + "  ".join(f"{s:>7}" for s in season_list))
print("─" * 52)
for name, pred in methods.items():
    row = f"{name:<16}"
    for s in season_list:
        sm = mask & (seasons_arr == s)
        row += f"  {rmse(pred, series, sm):7.4f}" if sm.sum() > 0 \
               else f"  {'N/A':>7}"
    print(row)

# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
timestamps    = full_idx.to_timestamp()
season_colors = {'Winter':"#b3d8ff",'Spring':"#b9fdb9",
                 'Summer':"#fceaad",'Autumn':"#fdc7b6"}

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# ── Panel 1: time series ──────────────────────────────────────────────────
ax = axes[0]

# Season shading (keep as is)
labeled = set()
for s, c in season_colors.items():
    idx    = np.where(seasons_arr == s)[0]
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    for b in np.split(idx, breaks):
        lbl = s if s not in labeled else ""
        ax.axvspan(timestamps[b[0]], timestamps[b[-1]],
                   alpha=0.18, color=c, label=lbl)
        labeled.add(s)

# True series — full line
ax.plot(timestamps, series, color='black', lw=2,
        label='True xCO₂', zorder=5)

# Masked points — show where gaps are
ax.scatter(timestamps[mask], series[mask],
           color='red', s=50, zorder=6, label='Masked (gap) points')

styles    = ['--', '-.', ':', '-']
pal       = ["#0073CBFF", '#414487FF','#22A884FF']

for (name, pred), ls, col in zip(methods.items(), styles, pal):
    # Full faint line — shows where each method runs
    ax.plot(timestamps, pred, ls, lw=1.0,
            color=col, alpha=0.3)
    # Bold dots only at masked positions — the actual imputed values
    ax.scatter(timestamps[mask], pred[mask],
               marker='o', s=35, color=col,
               label=name, zorder=7, alpha=0.9)

ax.set_title(f'CO₂ Imputation — {best_city}  ({len(series)} months)',
             fontsize=12)
ax.set_ylabel('xCO₂ (ppm)')
ax.legend(fontsize=8, ncol=4, loc='upper left')
fig.autofmt_xdate()

# ── Panel 2: RMSE by season ───────────────────────────────────────────────
ax2   = axes[1]
x_pos = np.arange(len(season_list))
w     = 0.2
cols  = ['#0073CBFF', '#414487FF','#22A884FF']

for i, (name, pred) in enumerate(methods.items()):
    vals = [rmse(pred, series, mask & (seasons_arr == s))
            if (mask & (seasons_arr == s)).sum() > 0 else 0
            for s in season_list]
    ax2.bar(x_pos + i * w, vals, w, label=name, color=cols[i], alpha=0.85)

ax2.set_xticks(x_pos + w * 1.5)
ax2.set_xticklabels(season_list)
ax2.set_ylabel('RMSE (ppm)')
ax2.set_title(f'Imputation Error by Season — {best_city}')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('co2_imputation_comparison_no_seasonal_encoding.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: co2_imputation_comparison_no_seasonal_encoding.png")
