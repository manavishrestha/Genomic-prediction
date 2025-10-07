# Genomic Prediction using Locus-specific Degree of Dominance Transformation

Manavi Shrestha

October 7, 2025

Dataset: Genomes To Fields (2025). Genomes to Fields 2024 Maize Genotype by Environment Prediction Competition. CyVerse Data Commons. DOI 10.25739/78mn-4394

## Data Preprocessing

### Trait data

Download the file:
   ```
   wget https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/1_Training_Trait_Data_2014_2023.csv
   ```
Run the file: Trait_data_processing.py\
File includes:
- Filtering for years 2019 to 2023
- Missing data removal
- Z-score outlier removal
> Output file: trait_clean_noOut.csv

### Genotype data
   ```
   wget https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt
   ```
Run the file: Genotype_preprocessing.py\
File includes:
- Removal of columns with high missing data
- knn-imputation
> Output file: Final_filtered_Recoded_imputed_SNP.csv

### Environmental covariate data
   ```
   wget https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/6_Training_EC_Data_2014_2023.csv
   ```
Run the file: Environmental covariates.py\
File includes
- Filtering for years 2019 to 2023
> Output file: Env_clean.csv

### Metadata
   ```
    wget https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/2_Training_Meta_Data_2014_2023.csv
   ```
Run the file: Metadata processing.py\
File includes:
- Filtering for years 2019 to 2023
- Removal of columns with high missing data
- Manual imputation of missing data
> Output file: metadata_clean_imputed.csv

### Weather data
   ```
   wget https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/4_Training_Weather_Data_2014_2023_full_year.csv
   ```
Run the file : Weather data.py\
File includes:
- Filtering for years 2019 to 2023
> Output file: weather_clean.csv

## BLUEs calculation
BLUEs are calulated separately for yield and plant height. Use the file trait_clean_noOut.csv as input for BLUEs.py\
The file BLUEs.py contains code for both traits. Run the file BLUEs.py

> Output file for yield: agg_BLUE_yield.csv\
> Output file for plant height: agg_BLUE_plant_height.csv







