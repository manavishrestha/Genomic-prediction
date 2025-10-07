import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
Final_dataset = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/Final_merged_March19.csv')
years = Final_dataset['Year'].values
unique_years = sorted(Final_dataset['Year'].unique())

# Specify features (X) and target (y)
X_all = Final_dataset.drop(columns=['Year', 'Treatment', 'Hybrid', 'Field_Location', 'Env', 'Average_BLUE'])
y = Final_dataset['Average_BLUE']

# Initialize XGBoost model
model = xgb.XGBRegressor(
    n_jobs=1,
    random_state=42
)

# Initialize scaler
scaler = StandardScaler()

# Extract SNP and Env columns
X = X_all.iloc[:, 0:2397]  # SNP columns
Env = X_all.iloc[:, 2397:]  # Env columns

results = []
y_true_all = []
y_pred_all = []
all_predictions = []

# Rolling year CV with dominance transformation
for i in range(len(unique_years) - 1):
    train_years = unique_years[:i + 1]
    test_year = unique_years[i + 1]

    train_index = Final_dataset['Year'].isin(train_years)
    test_index = Final_dataset['Year'] == test_year

    X_train, X_test = X.loc[train_index].copy(), X.loc[test_index].copy()
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    Env_train, Env_test = Env.loc[train_index].copy(), Env.loc[test_index].copy()

    # Dominance transformation
    snp_mean = np.zeros((X_train.shape[1], 3))

    for snp in range(X_train.shape[1]):
        for genotype in range(3):
            snp_mean[snp, genotype] = np.nanmean(y_train[X_train.iloc[:, snp] == genotype])
        snp_mean[np.isnan(snp_mean)] = 0.0

    my_choose = snp_mean[:, 0] > snp_mean[:, 2]

    temp = snp_mean[my_choose, 0].copy()
    snp_mean[my_choose, 0] = snp_mean[my_choose, 2]
    snp_mean[my_choose, 2] = temp

    X_train.iloc[:, my_choose] = np.abs(X_train.iloc[:, my_choose] - 2)
    X_test.iloc[:, my_choose] = np.abs(X_test.iloc[:, my_choose] - 2)

    d = (snp_mean[:, 1] - snp_mean[:, 0]) / (snp_mean[:, 2] - snp_mean[:, 0]) * 2
    d[np.isnan(d)] = 1
    d[d >= 2] = 2
    d[d <= 0] = 0

    for snp in range(X_train.shape[1]):
        het_train = X_train.iloc[:, snp] == 1
        X_train.iloc[het_train, snp] = d[snp]

    for snp in range(X_test.shape[1]):
        het_test = X_test.iloc[:, snp] == 1
        X_test.iloc[het_test, snp] = d[snp]

    X_train_final = pd.concat([X_train, Env_train], axis=1)
    X_test_final = pd.concat([X_test, Env_test], axis=1)

    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    # Train and predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Store results
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

    fold_df = pd.DataFrame({
        'Year': test_year,
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    all_predictions.append(fold_df)

    # Metrics (without R2)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    correlation, _ = pearsonr(y_test, y_pred)

    results.append({
        'Year': test_year,
        'RMSE': rmse,
        'Correlation': correlation
    })

# Pooled metrics
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

pooled_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
pooled_corr, _ = pearsonr(y_true_all, y_pred_all)

results.append({
    'Year': 'Pooled',
    'RMSE': pooled_rmse,
    'Correlation': pooled_corr
})

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('rolling_year_cv_results_yield_xgboost_transformed.csv', index=False)

# Save predictions
predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_df.to_csv('rolling_year_pred_yield_xgboost_transformed.csv', index=False)

# Save model
joblib.dump(model, 'transformed_xgboost_yield_rolling.pkl')
