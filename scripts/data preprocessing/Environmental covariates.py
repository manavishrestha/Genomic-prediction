import pandas as pd
# Import dataset
Environment_data_original = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/Training_data-20241213T092050Z-001/Training_data/6_Training_EC_Data_2014_2023.csv')
# Extract the year from the 'Env' column
Environment_data_original['Year'] = Environment_data_original['Env'].str.extract(r'_(\d{4})').astype(int)

# Filter the dataset for years between 2019 and 2023
Env_year = Environment_data_original[(Environment_data_original['Year'] >= 2019) & (Environment_data_original['Year'] <= 2023)]

# Drop the 'Year' column
Env_clean = Env_year.drop(columns=['Year'])

# Save the filtered dataset to a new CSV file
Env_clean.to_csv("/Users/manavi/Desktop/Code drafts/Env_clean.csv", index=False)
