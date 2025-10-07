# Trait data filtering
import pandas as pd
import numpy as np

# Import dataset
trait_data_original = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/Training_data-20241213T092050Z-001/Training_data/1_Training_Trait_Data_2014_2023.csv')
#Download the file 1_Training_Trait_Data_2014_2023.csv

# Inspect the dataset
trait_data_original.shape
trait_data_original.describe
trait_data_original.info

# Filtering for years 2019 to 2023
trait_2019_2023 = trait_data_original[(trait_data_original['Year'] >= 2019) & (trait_data_original['Year'] <= 2023)]
print("Data filtered for years 2019-2023:")
print(trait_2019_2023)

# Remove rows with mising value
trait_clean = trait_2019_2023.dropna()
print("Data after removing NA values:")
print(trait_clean)

#Check the dimension after removing missing values
trait_clean.shape

# Outlier removal using Z-score method

#Function to remove outliers from all columns simultaneously

def remove_outliers_multiple_columns(df, columns, z_thresh=3):
    z_scores = (df[columns] - df[columns].mean()) / df[columns].std()
    z_scores = z_scores.abs()
    # Rows with any Z-score >= threshold
    mask = (z_scores >= z_thresh).any(axis=1)
    outliers = df[mask]
    df_cleaned = df[~mask]
    return df_cleaned, outliers

## Select columns to remove the outliers
columns_to_check = trait_clean.columns[-10:].tolist()  # Last 10 columns
columns_to_check.append(trait_clean.columns[13])  # Add column 13 (index 12)

#Print the columns to be checked
print("Columns to check for outliers:", columns_to_check)

## Apply the function to remove outliers
trait_clean_noOut, outliers_df = remove_outliers_multiple_columns(trait_clean, columns_to_check)

#Inspect the clean dataset
trait_clean_noOut.shape

#Save the dataset to csv
trait_clean_noOut.to_csv('/Users/manavi/Desktop/Code drafts/trait_clean_noOut.csv', index = False)

