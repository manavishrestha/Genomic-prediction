# Merging for yield
import pandas as pd

#Importing datasets
Env_clean = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/Env_clean.csv')
weather_clean = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/weather_clean.csv')
metadata_clean = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/metadata_clean_imputed.csv')
Recoded_imputed_snp_data = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/March19_Final_filtered_Recoded_imputed_SNP.csv')
trait_aggregated_blues_yield = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/aggregated_blue_results_yield.csv')
trait_aggregated_blues_yield['Env'] = trait_aggregated_blues_yield['Field_Location'] + '_' + trait_aggregated_blues_yield['Year'].astype(str)

#checking unique environments before merging
Env_clean["Env"].nunique() #122 unique environment in Env clean
trait_aggregated_blues_yield["Env"].nunique() #77 unique env in trait
metadata_clean["Env"].nunique() #135 unique env in metadata
weather_clean["Env"].nunique() #135 unique env in weather clean

#Merging genotype and traits blues
genotype_blues = trait_aggregated_blues_yield.merge(Recoded_imputed_snp_data, on='Hybrid', how='inner')

#Merging with Env data
print(set(genotype_blues['Env']) - set(Env_clean['Env']))  # Missing in Env_clean
#Env 'ONH2_2019' is missing in Env clean.
#Removing the env from geno_blues to merge
genotype_blues = genotype_blues[genotype_blues['Env'] != 'ONH2_2019']
#Merge genotype_blues with Env_clean based on the 'Env' column
Env_geno_blues = genotype_blues.merge(Env_clean, on='Env', how='left')
Env_geno_blues.shape
#merging with metadata
print(set(Env_geno_blues['Env']) - set(metadata_clean['Env'])) #all present in Env geno blue is present in metadata
Env_geno_blues_meta = Env_geno_blues.merge(metadata_clean, on = 'Env', how = 'left')
#Merging with weather data
weather_agg = weather_clean.drop(columns=['Date']).groupby('Env').mean(numeric_only=True).reset_index()
print(set(Env_geno_blues_meta['Env']) - set(weather_agg['Env'])) #all present in Env_geno_blues_meta present in weather
Env_geno_blues_meta.shape
# Merge only matching 'Env' entries (inner join)
Env_geno_blues_meta_weather = Env_geno_blues_meta.merge(weather_agg, on='Env', how='inner')
Env_geno_blues_meta_weather.shape

#Save to csv
Env_geno_blues_meta_weather.to_csv("/Users/manavi/Desktop/Code drafts/Final_merged_March19.csv", index = False)

#Merging for plant height
import pandas as pd

#Import dataset
Env_clean = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/Env_clean.csv')
weather_clean = pd.read_csv(r'/Users/manavi/Desktop/thesis manavi/weather_clean.csv')
metadata_clean = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/metadata_clean_imputed.csv')
Recoded_imputed_snp_data = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/March19_Final_filtered_Recoded_imputed_SNP.csv')
trait_aggregated_blues_plant_height = pd.read_csv(r'/Users/manavi/Desktop/Pycharm/aggregated_blue_results_plant_height.csv')
trait_aggregated_blues_plant_height['Env'] = trait_aggregated_blues_plant_height['Field_Location'] + '_' + trait_aggregated_blues_plant_height['Year'].astype(str)

#Merging genotype and traits blues
genotype_blues = trait_aggregated_blues_plant_height.merge(Recoded_imputed_snp_data, on='Hybrid', how='inner')
print(set(genotype_blues['Env']) - set(Env_clean['Env']))  # Missing in Env_clean
#Env 'ONH2_2019' is missing in Env clean.
#Removing the env from geno_blues to merge
genotype_blues = genotype_blues[genotype_blues['Env'] != 'ONH2_2019']
#Merging with Env data
#Merge genotype_blues with Env_clean based on the 'Env' column
Env_geno_blues = genotype_blues.merge(Env_clean, on='Env', how='left')
#Merging with metadata
print(set(Env_geno_blues['Env']) - set(metadata_clean['Env'])) #all present in Env geno blue is present in metadata

Env_geno_blues_meta = Env_geno_blues.merge(metadata_clean, on = 'Env', how = 'left')

#merging with weather data
weather_agg = weather_clean.drop(columns=['Date']).groupby('Env').mean(numeric_only=True).reset_index()

print(set(Env_geno_blues_meta['Env']) - set(weather_agg['Env'])) #all present in Env_geno_blues_meta present in weather

# Merge only matching 'Env' entries (inner join)
Env_geno_blues_meta_weather_plant_height = Env_geno_blues_meta.merge(weather_agg, on='Env', how='inner')
Env_geno_blues_meta_weather_plant_height.to_csv("Final_merged_plant_height_March19.csv", index = False)

# Merging for plant height
#Download the BLUE for plant height obtained from BLUEs.py
#Repeat the same code as above
