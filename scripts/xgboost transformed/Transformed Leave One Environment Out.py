import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import joblib

# Fixing the random seed for reproducibility
np.random.seed(42)  # Fixing numpy's randomness
# Random state for XGBoost model
model = xgb.XGBRegressor(
    n_jobs=1,
    random_state=42  # Fixing randomness in XGBoost
)

# Load dataset
Final_dataset = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/Final_merged_March19.csv')
geno = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/March19_Final_filtered_Recoded_imputed_SNP.csv')

## Specify features (X) and target (y)
X_all = Final_dataset.drop(columns=['Year', 'Treatment', 'Hybrid', 'Field_Location', 'Env', 'Average_BLUE'])
y = Final_dataset['Average_BLUE']

# Get the unique field locations
field_locations = Final_dataset['Field_Location'].unique()

# Number of groups to create
n_groups = 5
# Initialize scaler
scaler = StandardScaler()

### Extract SNP columns
geno = geno.drop(columns="Hybrid")
geno.shape  # (5899, 2397)
X_all.columns[2395:2398]  # check index
X = X_all.iloc[:, 0:2397]  # Extract SNP columns
X.columns[-3:]  # verify last SNP columns
Env = X_all.iloc[:, 2397:]  # extract Env columns


# Create a function to randomly split locations into groups
def assign_locations_to_groups(field_locations, n_groups):
    np.random.shuffle(field_locations)  # Shuffle the field locations randomly
    group_assignments = np.array_split(field_locations, n_groups)  # Split the locations into n_groups

    # Create a dictionary for mapping field locations to group labels
    location_to_group = {}
    for group_index, group in enumerate(group_assignments):
        for loc in group:
            location_to_group[loc] = f'Group_{group_index + 1}'  # Assign a group label

    return location_to_group


# Create an empty list to store the results
all_results = []

# Nested loop for multiple iterations (if you want to repeat the group assignment multiple times)
num_iterations = 5  # How many times you want to repeat the group assignment and CV

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}:")

    # Randomly assign locations to groups for this iteration
    location_to_group = assign_locations_to_groups(field_locations, n_groups)

    # Print which locations are assigned to which group and save this info
    group_assignments_text = ""
    for group_num in range(1, n_groups + 1):
        locations_in_group = [loc for loc, group in location_to_group.items() if group == f'Group_{group_num}']
        group_assignments_text += f"Group {group_num}: {', '.join(locations_in_group)}\n"

    # Save the group assignments to a text file
    with open(f"iteration_{iteration + 1}_groups_tranfrom_yield__xgboost.txt", "w") as file:
        file.write(group_assignments_text)

    # Add the 'Group' column to Final_dataset based on location assignments
    Final_dataset['Group'] = Final_dataset['Field_Location'].map(location_to_group)

    # Initialize GroupKFold for cross-validation
    group_kfold = GroupKFold(n_splits=5)

    # Lists to store the metrics for all folds
    mse_scores = []
    rmse_scores = []
    correlation_scores = []

    # Nested loop for GroupKFold cross-validation
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, Final_dataset['Group'])):
        print(f"  Fold {fold + 1}:")

        # Get the group labels for train and test sets
        train_groups = Final_dataset.iloc[train_idx]['Group'].unique()
        test_groups = Final_dataset.iloc[test_idx]['Group'].unique()

        # Print which groups are used as train and test and save this info
        train_test_groups_text = f"Train groups: {', '.join(train_groups)}\nTest groups: {', '.join(test_groups)}\n"
        with open(f"iteration_{iteration + 1}_fold_{fold + 1}_train_test_groups_transform_yield_xgboost.txt", "w") as file:
            file.write(train_test_groups_text)

        # Split the data into training and test sets
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Dominance transformation
        Env_train = Env.iloc[train_idx].copy()
        Env_test = Env.iloc[test_idx].copy()

        snp_mean = np.zeros((X_train.shape[1], 3))

        # Loop over each SNP
        for snp in range(X_train.shape[1]):
            for genotype in range(3):
                snp_mean[snp, genotype] = np.nanmean(y_train[X_train.iloc[:, snp] == genotype])
            # Replace NaN values with 0
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

        # Replace missing values with 1
        d[np.isnan(d)] = 1

        # Boundaries for d
        d[d >= 2] = 2
        d[d <= 0] = 0

        # Weighted heterozygous genotypes
        # Weighted heterozygous genotypes for entire X (both train and test sets)
        # Apply dominance transformation to heterozygotes in training set
        for snp in range(X_train.shape[1]):
            het_train = X_train.iloc[:, snp] == 1
            X_train.iloc[het_train, snp] = d[snp]

        # Apply dominance transformation to heterozygotes in test set
        for snp in range(X_test.shape[1]):
            het_test = X_test.iloc[:, snp] == 1
            X_test.iloc[het_test, snp] = d[snp]

        # Add Env columns
        X_train_final = pd.concat([X_train, Env_train], axis=1)
        X_test_final = pd.concat([X_test, Env_test], axis=1)

        # apply to both training and testing set
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_test_scaled = scaler.transform(X_test_final)

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test_scaled)
        # Save predicted vs actual to a CSV file for each fold
        fold_results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        fold_results_df.to_csv(f"iteration_{iteration + 1}_fold_{fold + 1}_results.csv", index=False)
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        correlation, _ = pearsonr(y_test, y_pred)

        # Append the metrics for this fold
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        correlation_scores.append(correlation)

        # Print evaluation results for this fold
        print(f"    MSE: {mse:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    Pearson Correlation: {correlation:.4f}")
        print(f"    Test set actual values: {y_test.values}")
        print(f"    Test set predicted values: {y_pred}")
        print("-" * 50)

    # Compute pooled (overall) metrics after all folds in this iteration
    pooled_mse = np.mean(mse_scores)
    pooled_rmse = np.mean(rmse_scores)
    pooled_correlation = np.mean(correlation_scores)

    # Print the pooled cross-validation results for this iteration
    print("Pooled Cross-Validation Results:")
    print(f"Mean MSE: {pooled_mse:.4f}")
    print(f"Mean RMSE: {pooled_rmse:.4f}")
    print(f"Mean Pearson Correlation: {pooled_correlation:.4f}")
    print("=" * 50)

    # Create a DataFrame for this iteration's results
    iteration_results = pd.DataFrame({
        'Fold': [f"Fold {i + 1}" for i in range(5)],
        'MSE': mse_scores,
        'RMSE': rmse_scores,
        'Pearson Correlation': correlation_scores
    })

    # Add pooled row to the DataFrame
    iteration_results.loc['Pooled'] = [
        'Pooled',
        pooled_mse,
        pooled_rmse,
        pooled_correlation
    ]

    # Append to the all_results list (to store results for all iterations)
    all_results.append(iteration_results)

# Concatenate all iteration results into a single DataFrame
final_results = pd.concat(all_results, keys=[f"Iteration {i + 1}" for i in range(num_iterations)])

# Save the final results DataFrame to a CSV file
final_results.to_csv("Iteration_eval_results_transfor_yield_xgboost.csv")

# Optionally print the final results
print(final_results)
# Extract the pooled rows from the final_results DataFrame
pooled_results = final_results.xs('Pooled', level=1)  # Extracting rows labeled 'Pooled'

# Drop the 'Fold' column if it contains non-numeric values like 'Pooled' and keep only numeric columns
pooled_results_numeric = pooled_results.drop(columns='Fold')

# Compute the mean of each metric across all iterations
pooled_metrics_mean = pooled_results_numeric.mean()

# Create a DataFrame for the pooled metrics across all iterations
all_iteration_pooled_metrics = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'Pearson Correlation'],
    'Mean Across Iterations': [
        pooled_metrics_mean['MSE'],
        pooled_metrics_mean['RMSE'],
        pooled_metrics_mean['Pearson Correlation']
    ]
})

# Save the pooled metrics across all iterations to a CSV file
all_iteration_pooled_metrics.to_csv("transform_LOEO_yield_xgboost_all_iteration_metrics_pooled.csv", index=False)
print(all_iteration_pooled_metrics)

# Save the model for future use
joblib.dump(model, 'transform_LOEO_yieldxgboost_model_iteration.pkl')

print(all_iteration_pooled_metrics)
