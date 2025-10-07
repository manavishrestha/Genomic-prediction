import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

#BLUE's for yield

#Import cleaned trait file
#Import the file trait_clean_noOut.csv from Trait_data_processing.py
trait_clean_noOut = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/Training_data-20241213T092050Z-001/Training_data/trait_clean_noOut.csv')

#Create a copy of the original dataset
trait_clean_noOut_copy = trait_clean_noOut.copy()

#Placeholder for results
blue_results = []
coefficient_summaries = {}
aggregated_blue_results = []

## Loop over each year
for year in trait_clean_noOut_copy['Year'].unique():
    # Subset data for the current year
    year_data = trait_clean_noOut_copy[trait_clean_noOut_copy['Year'] == year].copy()

    # Fit fixed-effects model
    model = smf.ols('Yield_Mg_ha ~ C(Hybrid) + C(Replicate)', data=year_data).fit()

    # Print the model summary for the year
    print(f"Model Summary for Year {year}:\n")
    print(model.summary())

    # Save the model summary as a string
    model_summaries = model.summary().as_text()

    # Add predictions for each hybrid at each location (environment)
    year_data.loc[:, 'BLUE'] = model.predict(year_data)

    # Save BLUE results for this year
    for _, row in year_data.iterrows():
        blue_results.append({
            'Year': row['Year'],
            'Env': row['Env'],
            'Hybrid': row['Hybrid'],
            'Location': row['Field_Location'],
            'Replicate': row['Replicate'],
            'Yield_Mg_ha': row['Yield_Mg_ha'],
            'BLUE': row['BLUE']
        })

    # Extract coefficient table and create a DataFrame 
    coefficients = model.params.reset_index()
    coefficients.columns = ['Fixed Effect', 'Coefficient']
    coefficients['Interpretation'] = coefficients['Fixed Effect'].apply(
        lambda
            x: f"{x.split('[')[0]} yields {coefficients.loc[coefficients['Fixed Effect'] == x, 'Coefficient'].values[0]:.2f} more/less compared to the baseline."
        if 'C(' in x else f"Yield for baseline {x}."
    )
    coefficient_summaries[year] = coefficients

    # Calculate average BLUE for each Hybrid, Year, and Location
    aggregated_results = year_data.groupby(['Hybrid', 'Year', 'Field_Location']).agg(
        Average_BLUE=('BLUE', 'mean')
    ).reset_index()

    # Save to aggregated results
    aggregated_blue_results.append(aggregated_results)

# Combine results into final DataFrames
blue_results_df = pd.DataFrame(blue_results)
aggregated_blue_df = pd.concat(aggregated_blue_results).reset_index(drop=True)

# Save BLUE results and coefficient tables to CSV
aggregated_blue_df.to_csv('/Users/manavi/Desktop/Code drafts/agg_BLUE_yield.csv', index=False)

# BLUE's for plant height
trait_clean_noOut_plant_height = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/Training_data-20241213T092050Z-001/Training_data/trait_clean_noOut.csv')
trait_clean_noOut_plant_height = trait_clean_noOut.copy()

#Placeholder for plant height results
blue_results_plant_height = []
coefficient_summaries_plant_height = {}
aggregated_blue_results_plant_height = []

# Loop over each year for plant height analysis
for year in trait_clean_noOut_plant_height['Year'].unique():
    # Subset data for the current year
    year_data_plant_height = trait_clean_noOut_plant_height[trait_clean_noOut_plant_height['Year'] == year].copy()

    # Fit fixed-effects model for plant height
    model_plant_height = smf.ols('Plant_Height_cm ~ C(Hybrid) + C(Replicate)', data=year_data_plant_height).fit()

    # Print the model summary for plant height in the current year
    print(f"Model Summary for Plant Height - Year {year}:\n")
    print(model_plant_height.summary())

    # Save the model summary as a string
    model_summaries_plant_height = model_plant_height.summary().as_text()

    # Add predictions for each hybrid at each location (environment) for plant height
    year_data_plant_height.loc[:, 'BLUE_Plant_Height'] = model_plant_height.predict(year_data_plant_height)

    # Save BLUE results for this year for plant height
    for _, row in year_data_plant_height.iterrows():
        blue_results_plant_height.append({
            'Year': row['Year'],
            'Env': row['Env'],
            'Hybrid': row['Hybrid'],
            'Location': row['Field_Location'],
            'Replicate': row['Replicate'],
            'Plant_Height_cm': row['Plant_Height_cm'],
            'BLUE_Plant_Height': row['BLUE_Plant_Height']
        })

    # Extract coefficient table for plant height and create a DataFrame (optional)
    coefficients_plant_height = model_plant_height.params.reset_index()
    coefficients_plant_height.columns = ['Fixed Effect', 'Coefficient']
    coefficients_plant_height['Interpretation'] = coefficients_plant_height['Fixed Effect'].apply(
        lambda
            x: f"{x.split('[')[0]} affects plant height by {coefficients_plant_height.loc[coefficients_plant_height['Fixed Effect'] == x, 'Coefficient'].values[0]:.2f} compared to the baseline."
        if 'C(' in x else f"Baseline effect for {x}."
    )
    coefficient_summaries_plant_height[year] = coefficients_plant_height

    # Calculate average BLUE for plant height for each Hybrid, Year, and Location
    aggregated_results_plant_height = year_data_plant_height.groupby(['Hybrid', 'Year', 'Field_Location']).agg(
        Average_BLUE_Plant_Height=('BLUE_Plant_Height', 'mean')
    ).reset_index()

    # Save to aggregated results for plant height
    aggregated_blue_results_plant_height.append(aggregated_results_plant_height)

# Combine results for plant height into final DataFrames
blue_results_df_plant_height = pd.DataFrame(blue_results_plant_height)
aggregated_blue_df_plant_height = pd.concat(aggregated_blue_results_plant_height).reset_index(drop=True)

aggregated_blue_df_plant_height.describe()

# Save BLUE results and coefficient tables to CSV for plant height
blue_results_df_plant_height.to_csv('/Users/manavi/Desktop/Code drafts/agg_BLUE_plant_height.csv', index=False)
