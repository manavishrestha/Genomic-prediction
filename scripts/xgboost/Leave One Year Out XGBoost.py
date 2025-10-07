import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import pearsonr

# Load dataset
Final_dataset = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/Final_merged_March19.csv')
years = Final_dataset['Year'].values  # For grouping

# Specify features (X) and target (y)
X = Final_dataset.drop(columns=['Year', 'Treatment','Hybrid', 'Field_Location', 'Env', 'Average_BLUE'])
y = Final_dataset['Average_BLUE']

# Initialize XGBoost model with optimized parameters
model = xgb.XGBRegressor(
    n_jobs=1,
    random_state=42  # For reproducibility
)

# Leave-One-Year-Out CV
leave_one_year_out = LeaveOneGroupOut()
results = []
all_y_test = []
all_y_pred = []
all_hybrid_env = []

for train_idx, test_idx in leave_one_year_out.split(X, y, groups=years):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get hybrid_env for test samples
    hybrid_env = Final_dataset.iloc[test_idx][['Hybrid', 'Env']].agg('_'.join, axis=1).values

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    corr, _ = pearsonr(y_test, y_pred)  # Pearson correlation

    # Store results
    test_year = years[test_idx][0]
    results.append({
        'Year': test_year,
        'RMSE': rmse,
        'Correlation': corr
    })

    # Print yearly performance
    print(f"Year {test_year}:")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  Pearson r = {corr:.3f}\n")

    # Save fold predictions
    fold_df = pd.DataFrame({
        'Hybrid_Env': hybrid_env,
        'Actual': y_test,
        'Predicted': y_pred
    })
    # Save to CSV per fold
    fold_df.to_csv(f'fold_{test_year}_yield_xgboost_LOYO.csv', index=False)

    # Append for final predictions
    all_y_test.append(y_test)
    all_y_pred.append(y_pred)
    all_hybrid_env.append(hybrid_env)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate overall means and standard deviations
mean_rmse = results_df['RMSE'].mean()
mean_corr = results_df['Correlation'].mean()

# Append pooled statistics as the last row
pooled_row = pd.DataFrame([{
    'Year': 'Pooled',
    'RMSE': mean_rmse,
    'Correlation': mean_corr
}])
results_df = pd.concat([results_df, pooled_row], ignore_index=True)

# Save results to CSV
results_df.to_csv('leave_one_year_out_results_yield_xgboost.csv', index=False)

# Concatenate all actual and predicted values
final_y_true = np.concatenate(all_y_test)
final_y_pred = np.concatenate(all_y_pred)
final_hybrid_env = np.concatenate(all_hybrid_env)

# Create final DataFrame with predictions
final_predictions_df = pd.DataFrame({
    'Hybrid_Env': final_hybrid_env,
    'Actual': final_y_true,
    'Predicted': final_y_pred
})

# Save final predictions
final_predictions_df.to_csv("LOYO_final_predictions_xgboost_yield.csv", index=False)

# Leave One Year Out XGBoost plant height

# Load dataset Final merged for plant height
Final_dataset = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/Final_merged_plant_height_March19.csv')
years = Final_dataset['Year'].values  # For grouping

# Specify features (X) and target (y)
X = Final_dataset.drop(columns=['Year', 'Treatment', 'Hybrid', 'Field_Location', 'Env', 'Average_BLUE_Plant_Height'])
y = Final_dataset['Average_BLUE_Plant_Height']

# Repeat the process from model initialization

