import pandas as pd

# Import weather data
weather_data_original = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/Training_data-20241213T092050Z-001/Training_data/4_Training_Weather_Data_2014_2023_full_year.csv')

#Extract the year from the 'Env' column
weather_data_original['Year'] = weather_data_original['Env'].str.extract(r'_(\d{4})').astype(int)

#Filter the dataset for years between 2019 and 2023
weather_year = weather_data_original[(weather_data_original['Year'] >= 2019) & (weather_data_original['Year'] <= 2023)]

#Drop the 'Year' column
weather_clean = weather_year.drop(columns=['Year'])

#Save the filtered dataset to a new CSV file
weather_clean.to_csv("/Users/manavi/Desktop/Code drafts/weather_clean.csv", index=False)