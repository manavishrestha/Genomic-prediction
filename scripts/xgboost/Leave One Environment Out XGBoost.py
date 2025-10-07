# Leave One Env Out xgboost yield - April 12
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold

#Fixing the random seed for reproducibility
np.random.seed(42)

## Random state for XGBoost model
model = xgb.XGBRegressor(
    n_jobs=1,
    random_state=42
)

## Load dataset
Final_dataset = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/Final_merged_March19.csv')

## Specify features (X) and target (y)
X = Final_dataset.drop(columns=['Year', 'Treatment', 'Hybrid', 'Field_Location', 'Env', 'Average_BLUE'])
y = Final_dataset['Average_BLUE']

## Get the unique field locations
field_locations = Final_dataset['Field_Location'].unique()

## Number of groups to create
n_groups = 5

## Create a function to randomly split locations into groups
def assign_locations_to_groups(field_locations, n_groups):
    np.random.shuffle(field_locations)  # Shuffle the field locations randomly
    group_assignments = np.array_split(field_locations, n_groups)  # Split the locations into n_groups

    # Create a dictionary for mapping field locations to group labels
    location_to_group = {}
    for group_index, group in enumerate(group_assignments):
        for loc in group:
            location_to_group[loc] = f'Group_{group_index + 1}'  # Assign a group label

    return location_to_group


## Create an empty list to store the results
all_results = []

## Nested loop for multiple iterations
num_iterations = 5

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
    with open(f"iteration_{iteration + 1}_groups.txt", "w") as file:
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
        with open(f"iteration_{iteration + 1}_fold_{fold + 1}_train_test_groups.txt", "w") as file:
            file.write(train_test_groups_text)

        # Split the data into training and test sets
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

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

## Concatenate all iteration results into a single DataFrame
final_results = pd.concat(all_results, keys=[f"Iteration {i + 1}" for i in range(num_iterations)])

## Save the final results DataFrame to a CSV file
final_results.to_csv("Iteration_eval_results.csv")

## Optionally print the final results
print(final_results)

#Fixing the random seed for reproducibility
np.random.seed(42)

## Random state for XGBoost model
model = xgb.XGBRegressor(
    n_jobs=1,
    random_state=42
)

## Load dataset
Final_dataset = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/Final_merged_March19.csv')

## Specify features (X) and target (y)
X = Final_dataset.drop(columns=['Year', 'Treatment', 'Hybrid', 'Field_Location', 'Env', 'Average_BLUE_Plant_Height'])
y = Final_dataset['Average_BLUE_Plant_Height']

#Repeat the steps after getting unique field location
