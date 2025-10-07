# Genomic Prediction using Locus-specific Degree of Dominance Transformation

Manavi Shrestha

October 7, 2025

Dataset: Genomes To Fields (2025). Genomes to Fields 2024 Maize Genotype by Environment Prediction Competition. CyVerse Data Commons. DOI 10.25739/78mn-4394

## Download the datasets

1. Trait data
   ```
   wget https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/1_Training_Trait_Data_2014_2023.csv
   ```
> Output file: trait_clean_noOut.csv
2. Genotype data
   ```
   wget https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt
   ```
> Output file: Final_filtered_Recoded_imputed_SNP.csv
3. Environmental covariate data
   ```
   wget https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/6_Training_EC_Data_2014_2023.csv
   ```
> Output file: Env_clean.csv
4. Metadata
   ```
 wget https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/2_Training_Meta_Data_2014_2023.csv
   ```
> Output file: metadata_clean_imputed.csv
5. Weather data
   ```
wget https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data/4_Training_Weather_Data_2014_2023_full_year.csv
   ```
> Output file: weather_clean.csv




