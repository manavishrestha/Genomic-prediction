import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

### Import Datasets
Final_dataset = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/Final_merged_March19.csv')
years = Final_dataset['Year'].values  # For grouping
geno = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/March19_Final_filtered_Recoded_imputed_SNP.csv')

## Specify features (X) and target (y)
X_all = Final_dataset.drop(columns=['Year', 'Treatment', 'Hybrid', 'Field_Location', 'Env', 'Average_BLUE'])
y = Final_dataset['Average_BLUE']

### Extract SNP columns
geno = geno.drop(columns="Hybrid")
X = X_all.iloc[:, 0:2397]  # Extract SNP columns
Env = X_all.iloc[:, 2397:]  # extract Env columns

# Initialize model
model = xgb.XGBRegressor(
    n_jobs=1,
    random_state=22
)

# Initialize scaler
scaler = StandardScaler()

# Leave-One-Year-Out CV
leave_one_year_out = LeaveOneGroupOut()
results = []
all_y_test = []
all_y_pred = []
all_hybrid_env = []

for train_idx, test_idx in leave_one_year_out.split(X, y, groups=years):
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    Env_train = Env.iloc[train_idx]
    Env_test = Env.iloc[test_idx]

    snp_mean = np.zeros((X_train.shape[1], 3))

    # Loop over each SNP
    for snp in range(X_train.shape[1]):
        for genotype in range(3):
            snp_mean[snp, genotype] = np.nanmean(y_train[X_train.iloc[:, snp] == genotype])
        snp_mean[np.isnan(snp_mean)] = 0.0

    # Create a condition to select rows where snpMean[:, 0] > snpMean[:, 2]
    my_choose = snp_mean[:, 0] > snp_mean[:, 2]

    # Swap values where needed
    temp = snp_mean[my_choose, 0].copy()
    snp_mean[my_choose, 0] = snp_mean[my_choose, 2]
    snp_mean[my_choose, 2] = temp

    # Flip genotypes accordingly in training and test set
    X_train.iloc[:, my_choose] = np.abs(X_train.iloc[:, my_choose] - 2)
    X_test.iloc[:, my_choose] = np.abs(X_test.iloc[:, my_choose] - 2)

    # Calculate d (degree of dominance)
    d = (snp_mean[:, 1] - snp_mean[:, 0]) / (snp_mean[:, 2] - snp_mean[:, 0]) * 2
    d[np.isnan(d)] = 1
    d[d >= 2] = 2
    d[d <= 0] = 0

    # Weighted heterozygous genotypes for entire X (both train and test sets)
    for snp in range(X_train.shape[1]):
        het_train = X_train.iloc[:, snp] == 1
        X_train.iloc[het_train, snp] = d[snp]

    for snp in range(X_test.shape[1]):
        het_test = X_test.iloc[:, snp] == 1
        X_test.iloc[het_test, snp] = d[snp]

    # Add Env columns
    X_train_final = pd.concat([X_train, Env_train], axis=1)
    X_test_final = pd.concat([X_test, Env_test], axis=1)

    # apply to both training and testing set
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    # Train and predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Get hybrid_env for test samples
    hybrid_env = Final_dataset.iloc[test_idx][['Hybrid', 'Env']].agg('_'.join, axis=1).values

    # Calculate metrics (without R2)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    corr, _ = pearsonr(y_test, y_pred)

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
    fold_df.to_csv(f'1_fold_{test_year}_transformed_yield_xgboost_LOYO.csv', index=False)

    all_y_test.append(y_test)
    all_y_pred.append(y_pred)
    all_hybrid_env.append(hybrid_env)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate overall means
mean_rmse = results_df['RMSE'].mean()
mean_corr = results_df['Correlation'].mean()

# Append pooled statistics as the last row
pooled_row = pd.DataFrame([{
    'Year': 'Pooled',
    'RMSE': mean_rmse,
    'Correlation': mean_corr
}])
results_df = pd.concat([results_df, pooled_row], ignore_index=True)
results_df.to_csv('1_transformed_loyo_yiled_xgboost.csv', index=False)

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
final_predictions_df.to_csv("1_transformed_LOYO_final_predictions_xgboost_yield.csv", index=False)

# Change these steps for plant height and repeat
# X_all = Final_dataset.drop(columns=['Year', 'Treatment', 'Hybrid', 'Field_Location', 'Env', 'Average_BLUE_Plant_Height'])
# y = Final_dataset['Average_BLUE_Plant_Height']