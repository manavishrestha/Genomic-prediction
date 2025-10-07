import pandas as pd
import datatable as dt
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
import numpy as np
import random
from sklearn.utils import check_random_state
#Import genotype data
genotype_original = dt.fread(r'/Users/manavi/Desktop/thesis manavi/Training_data-20241213T092050Z-001/Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt')
genotype_data = genotype_original.to_pandas()

# Calculate missing value percentages only for columns that have missing values
missing_percentage = (genotype_data.isnull().sum() / len(genotype_data)) * 100
missing_percentage = missing_percentage[missing_percentage > 0]  # Filter only columns with missing values
missing_percentage = missing_percentage.round(2)  # Round to 2 decimal places
print("Missing value percentages:")
print(missing_percentage)

# Identify columns with missing percentage >= 50%
high_missing_columns = missing_percentage[missing_percentage >= 50].index.tolist()
print("\nColumns to be removed (≥50% missing values):")
print(high_missing_columns)

# Make a copy of the dataframe and remove high-missing columns
genotype_missing_columns_removed = genotype_data.drop(columns=high_missing_columns)

# Print shape to confirm removal
print("\nShape before removal:", genotype_data.shape) #(5900, 2426)
print("Shape after removal:", genotype_missing_columns_removed.shape) #28 columns removed #(5900, 2398)

# Ensure the first row is treated as column names correctly
genotype_missing_columns_removed.columns = genotype_missing_columns_removed.iloc[0]  # Assign first row as column names
genotype_missing_columns_removed = genotype_missing_columns_removed[1:].reset_index(drop=True)  # Drop the first row and reset index
print(genotype_missing_columns_removed.iloc[0,0])

# Ensure the first column is the hybrid names
hybrids = genotype_missing_columns_removed.iloc[:, 0].reset_index(drop=True)  # Extract hybrids and reset index
snp_data = genotype_missing_columns_removed.iloc[:, 1:] # Exclude first column (hybrid names)

# Convert SNP data to numeric (force non-numeric to NaN)
snp_data = snp_data.apply(pd.to_numeric, errors='coerce')

# Set the random seed to ensure reproducibility
np.random.seed(42)  # Set seed for numpy
random.seed(42)  # Set seed for python random module

# Set random seed for sklearn
random_state = check_random_state(42)

# Function for knn imputation
def knn_impute_mode(snp_data, n_neighbors=5, placeholder=-1):
    snp_data_copy = snp_data.copy()

    # Replace NaNs with a placeholder (-1) for fitting the model
    snp_data_filled = snp_data.fillna(placeholder)

    # Fit Nearest Neighbors model using nan_euclidean distance
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='nan_euclidean')
    nn.fit(snp_data_filled)

    # Find nearest neighbors
    distances, indices = nn.kneighbors(snp_data_filled)

    # Iterate over missing values
    for i in range(snp_data.shape[0]):  # Iterate over rows (samples)
        for j in range(snp_data.shape[1]):  # Iterate over columns (SNPs)
            if pd.isna(snp_data.iloc[i, j]):  # Check if value is missing
                neighbor_values = snp_data.iloc[indices[i], j].dropna().values  # Get neighbor SNP values

                if len(neighbor_values) > 0:
                    # Compute mode of the neighbors
                    imputed_value = mode(neighbor_values, keepdims=True).mode[0]

                    # Ensure imputed value is 0, 0.5, or 1
                    if imputed_value in [0, 0.5, 1]:
                        snp_data_copy.iloc[i, j] = imputed_value
                    else:
                        snp_data_copy.iloc[i, j] = snp_data.iloc[:, j].mode()[0]  # Fallback to column mode
                else:
                    # If no valid neighbors, use column mode
                    snp_data_copy.iloc[i, j] = snp_data.iloc[:, j].mode()[0]

    return snp_data_copy

# Apply k-NN mode imputation

snp_data_imputed = knn_impute_mode(snp_data)

# Function to check if any values are outside of {0, 0.5, 1}
def check_invalid_values(df):
    """
    Check if there are any values in the dataframe outside of {0, 0.5, 1}.
    """
    invalid_values = df[~df.isin([0, 0.5, 1])].stack()
    if not invalid_values.empty:
        print(f"Warning: Found {invalid_values.shape[0]} invalid values:")
        print(invalid_values)
    else:
        print("All values are valid (0, 0.5, 1).")

# Check the imputed dataframe for invalid values
check_invalid_values(snp_data_imputed)

# Reattach the hybrids column back to the imputed SNP data
snp_data_imputed.insert(0, 'Hybrid', hybrids)
# Save the imputed SNP data to a new file
snp_data_imputed.to_csv("Final_imputed_genotype.csv", index = False)


# Recoding
# Make a copy to modify
imputed_snp_data_copy = snp_data_imputed.copy()

# Exclude the first column (hybrid names)
snp_columns = imputed_snp_data_copy.columns[1:]  # All columns except the first

# Convert the SNP columns to numeric, keeping the first column unchanged
imputed_snp_data_copy[snp_columns] = imputed_snp_data_copy[snp_columns].apply(pd.to_numeric, errors='coerce')

# Replace 0.5 → 1 and 1 → 2 in the SNP columns
imputed_snp_data_copy[snp_columns] = imputed_snp_data_copy[snp_columns].replace({0.5: 1, 1: 2})

# Check the first few rows to confirm changes
print(imputed_snp_data_copy.head())

# Function to cheack if all values are between 0 , 1 and 2
def check_valid_values(df):
    # Exclude the first column (hybrid names) and check for values not in [0, 1, 2] in the rest of the dataframe
    snp_data = df.iloc[:, 1:]  # Exclude the first column

    # Check for invalid values
    invalid_values = snp_data[~snp_data.isin([0, 1, 2])].stack()  # Check for values not in [0, 1, 2]

    if not invalid_values.empty:
        print(f"Warning: Found {invalid_values.shape[0]} invalid values:")
        print(invalid_values)
    else:
        print("All values are valid (0, 1, 2).")


# Check for invalid values
check_valid_values(imputed_snp_data_copy)

# Save the modified dataframe
imputed_snp_data_copy.to_csv("Final_filtered_Recoded_imputed_SNP.csv", index = False)


