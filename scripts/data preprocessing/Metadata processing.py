import pandas as pd

# Import dataset
meta_data_original = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/Training_data-20241213T092050Z-001/Training_data/2_Training_Meta_Data_2014_2023.csv')

# Filter for years
metadata_clean = meta_data_original[(meta_data_original['Year'] >= 2019) & (meta_data_original['Year'] <= 2023)]

#Save the filtered dataset to a new CSV file
metadata_clean.to_csv("metadata_clean.csv", index=False)
metadata_clean.shape

# Remove columns with high missing values.
#Calculate missing value percentages only for columns that have missing values
missing_percentage = (metadata_clean.isnull().sum() / len(metadata_clean)) * 100
missing_percentage = missing_percentage[missing_percentage > 0]  # Filter only columns with missing values
missing_percentage = missing_percentage.round(2)  # Round to 2 decimal places
print(missing_percentage)

#Filter only columns where missing percentage is >= 50%
high_missing_columns = missing_percentage[missing_percentage >= 50].round(2)
print("Columns with 50% or more missing values:")
print(high_missing_columns)

#Get column names with >50% missing values
columns_with_high_missing = high_missing_columns.index.tolist()

#Remove columns with >50% missing values from metadata_clean
metadata_clean = metadata_clean.drop(columns=columns_with_high_missing)

#check shape after removal
metadata_clean.shape

# Selection of columns to keep.
df_col_to_keep = [
    'Env', 'Experiment_Code', 'Treatment',
    'Latitude_of_Field_Corner_#1 (lower left)',
    'Latitude_of_Field_Corner_#2 (lower right)',
    'Latitude_of_Field_Corner_#3 (upper right)',
    'Latitude_of_Field_Corner_#4 (upper left)',
    'Longitude_of_Field_Corner_#1 (lower left)',
    'Longitude_of_Field_Corner_#2 (lower right)',
    'Longitude_of_Field_Corner_#3 (upper right)',
    'Longitude_of_Field_Corner_#4 (upper left)'
]

# Keep only specified columns
metadata_clean = metadata_clean.loc[:, df_col_to_keep]

# Imputing latitudes

#Environment-Based Field Coordinates Calculation
#Compute the mean latitude and longitude for each environment group
metadata_clean['Field_lat'] = metadata_clean.groupby('Env')[
    ['Latitude_of_Field_Corner_#1 (lower left)',
     'Latitude_of_Field_Corner_#2 (lower right)',
     'Latitude_of_Field_Corner_#3 (upper right)',
     'Latitude_of_Field_Corner_#4 (upper left)']
].transform(lambda x: x.mean(skipna=True)).iloc[:, 0]

metadata_clean['Field_long'] = metadata_clean.groupby('Env')[
    ['Longitude_of_Field_Corner_#1 (lower left)',
     'Longitude_of_Field_Corner_#2 (lower right)',
     'Longitude_of_Field_Corner_#3 (upper right)',
     'Longitude_of_Field_Corner_#4 (upper left)']
].transform(lambda x: x.mean(skipna=True)).iloc[:, 0]

#Drop the individual corner coordinate columns
metadata_clean.drop(columns=[
    'Latitude_of_Field_Corner_#1 (lower left)', 'Latitude_of_Field_Corner_#2 (lower right)',
    'Latitude_of_Field_Corner_#3 (upper right)', 'Latitude_of_Field_Corner_#4 (upper left)',
    'Longitude_of_Field_Corner_#1 (lower left)', 'Longitude_of_Field_Corner_#2 (lower right)',
    'Longitude_of_Field_Corner_#3 (upper right)', 'Longitude_of_Field_Corner_#4 (upper left)'
], inplace=True)

metadata_clean.shape


#Extract rows where Field_lat or Field_long is missing (NaN)
missing_rows = metadata_clean[metadata_clean[['Field_lat', 'Field_long']].isna().any(axis=1)]
print(missing_rows)

# Extract unique Experiment_Code values from missing rows
missing_experiment_codes = missing_rows['Experiment_Code'].unique()

# Extract all rows with the same Experiment_Code as those in missing_rows
grouped_rows = metadata_clean[metadata_clean['Experiment_Code'].isin(missing_experiment_codes)]

# Sort the grouped rows by Experiment_Code to keep them together
grouped_rows = grouped_rows.sort_values(by='Experiment_Code')

# Display the grouped rows
print(grouped_rows)

##TX4_2019 not present i the BLUE datset will be removed during merging.(not imputed)

# Imputing each Env individually
# Handling Missing Longitude Data for "ILH1_2021"
ILH1_mean_long = metadata_clean.loc[
    (metadata_clean['Experiment_Code'] == 'ILH1') & (metadata_clean['Env'] != 'ILH1_2021'),
    'Field_long'
].mean(skipna=True)

## Handling Missing Latitude Data for "ILH1_2021"
ILH1_mean_lat = metadata_clean.loc[
    (metadata_clean['Experiment_Code'] == 'ILH1') & (metadata_clean['Env'] != 'ILH1_2021'),
    'Field_lat'
].mean(skipna=True)

# Update missing values for ILH1_2021
metadata_clean.loc[metadata_clean['Env'] == 'ILH1_2021', 'Field_long'] = ILH1_mean_long
metadata_clean.loc[metadata_clean['Env'] == 'ILH1_2021', 'Field_lat'] = ILH1_mean_lat

## Handling Missing Longitude Data for "NEH2_2019"
NEH2_2019_mean_long = metadata_clean.loc[
    (metadata_clean['Experiment_Code'] == 'NEH2') & (metadata_clean['Env'] != 'NEH2_2019') & (metadata_clean['Env'] != 'NEH2_2020'),
    'Field_long'
].mean(skipna=True)
## Handling Missing Latitude Data for "NEH2_2019"
NEH2_2019_mean_lat = metadata_clean.loc[
    (metadata_clean['Experiment_Code'] == 'ILH1') & (metadata_clean['Env'] != 'ILH1_2021') & (metadata_clean['Env'] != 'NEH2_2020'),
    'Field_lat'
].mean(skipna=True)

# Update missing values for NEH2_2019
metadata_clean.loc[metadata_clean['Env'] == 'NEH2_2019', 'Field_long'] = NEH2_2019_mean_long
metadata_clean.loc[metadata_clean['Env'] == 'NEH2_2019', 'Field_lat'] = NEH2_2019_mean_lat

# Handling Missing Latitude and Longitude Data for "NEH2_2020" by using values from "NEH2_2019"
metadata_clean.loc[metadata_clean['Env'] == 'NEH2_2020', 'Field_long'] = NEH2_2019_mean_long
metadata_clean.loc[metadata_clean['Env'] == 'NEH2_2020', 'Field_lat'] = NEH2_2019_mean_lat

## Handling Missing Longitude Data for "NEH3_2020"
NEH3_2020_mean_long = metadata_clean.loc[
    (metadata_clean['Experiment_Code'] == 'NEH3') & (metadata_clean['Env'] != 'NEH3_2020'),
    'Field_long'
].mean(skipna=True)

## Handling Missing Latitude Data for "NEH3_2020"
NEH3_2020_mean_lat = metadata_clean.loc[
    (metadata_clean['Experiment_Code'] == 'NEH3') & (metadata_clean['Env'] != 'NEH3_2020'),
    'Field_lat'
].mean(skipna=True)

# Update missing values for NEH3_2020
metadata_clean.loc[metadata_clean['Env'] == 'NEH3_2020', 'Field_long'] =  NEH3_2020_mean_long
metadata_clean.loc[metadata_clean['Env'] == 'NEH3_2020', 'Field_lat'] =  NEH3_2020_mean_lat

## Handling Missing Longitude Data for "NYS1_2021"
NYS1_2021_mean_long = metadata_clean.loc[
    (metadata_clean['Experiment_Code'] == 'NYS1') & (metadata_clean['Env'] != 'NYS1_2021'),
    'Field_long'
].mean(skipna=True)

## Handling Missing Latitude Data for "NYS1_2021"
NYS1_2021_mean_lat = metadata_clean.loc[
    (metadata_clean['Experiment_Code'] == 'NYS1') & (metadata_clean['Env'] != 'NYS1_2021'),
    'Field_lat'
].mean(skipna=True)

# Handling Missing Latitude and Longitude Data for "NYS1_2021"
metadata_clean.loc[metadata_clean['Env'] == 'NYS1_2021', 'Field_long'] = NYS1_2021_mean_long
metadata_clean.loc[metadata_clean['Env'] == 'NYS1_2021', 'Field_lat'] = NYS1_2021_mean_lat

#missing values
# Count missing values per column
missing_values = metadata_clean.isnull().sum() #only 1 for TX4_2019
#summary
metadata_clean.describe

# Save to csv
metadata_clean.to_csv("/Users/manavi/Desktop/Code drafts/metadata_clean_imputed.csv", index=False)

