import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


#Rolling Year Cross validation for Yield
# Load dataset
Final_dataset = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/Final_merged_March19.csv')
years = Final_dataset['Year'].values
unique_years = sorted(Final_dataset['Year'].unique())

# Specify features (X) and target (y)
X = Final_dataset.drop(columns=['Year', 'Treatment', 'Hybrid', 'Field_Location', 'Env', 'Average_BLUE'])
y = Final_dataset['Average_BLUE']

# Initialize XGBoost model
model = xgb.XGBRegressor(
    n_jobs=1,
    random_state=42
)

results = []
y_true_all = []
y_pred_all = []
# Create a list to store predictions per year
all_predictions = []
# Rolling year CV
for i in range(len(unique_years) - 1):
    train_years = unique_years[:i + 1]
    test_year = unique_years[i + 1]

    train_index = Final_dataset['Year'].isin(train_years)
    test_index = Final_dataset['Year'] == test_year

    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Store for pooled metrics
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

    # Store predictions for each test sample
    fold_df = pd.DataFrame({
        'Year': test_year,
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    all_predictions.append(fold_df)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    correlation, _ = pearsonr(y_test, y_pred)

    results.append({
        'Year': test_year,
        'RMSE': rmse,
        'Correlation': correlation
    })

# Save results
results_df = pd.DataFrame(results)

# Calculate mean RMSE and Correlation across folds
mean_rmse = results_df['RMSE'].mean()
mean_corr = results_df['Correlation'].mean()

# Append pooled statistics as the last row
pooled_row = pd.DataFrame([{
    'Year': 'Pooled',
    'RMSE': mean_rmse,
    'Correlation': mean_corr
}])
results_df = pd.concat([results_df, pooled_row], ignore_index=True)

#results_df.to_csv('rolling_year_cv_results_yield_xgboost_no_opt.csv', index=False)

# Concatenate all folds' predictions
predictions_df = pd.concat(all_predictions, ignore_index=True)

# Save to CSV
predictions_df.to_csv('rolling_year_pred_yield_xgboost.csv', index=False)

#Rolling Year cross validation for plant height

# Load dataset
Final_dataset = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/Final_merged_March19.csv')
years = Final_dataset['Year'].values
unique_years = sorted(Final_dataset['Year'].unique())

# Specify features (X) and target (y)
X = Final_dataset.drop(columns=['Year', 'Treatment', 'Hybrid', 'Field_Location', 'Env', 'Average_BLUE_Plant_Height'])
y = Final_dataset['Average_BLUE_Plant_Height']

# Repeat the process from model initialization

