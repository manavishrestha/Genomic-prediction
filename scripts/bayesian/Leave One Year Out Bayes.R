# Leave-one year out CV ##########--15 April

library(BGLR)

# Import datasets
genotype_data <- read.csv("March19_Final_filtered_Recoded_imputed_SNP.csv")
Final_merged <- read.csv("Final_merged_March19.csv")
env_data <- read.csv("Env_clean.csv")
meta_data <- read.csv("metadata_clean_imputed.csv")
weather_data <- read.csv("weather_clean.csv")

# As separate dataframes for separate matrices
blues <- Final_merged[, 1:5]
geno <- Final_merged[, 6:2402]   # all numeric
env  <- Final_merged[, 2403:3056]
meta <- Final_merged[, 3058:3059] # Exclude treatment
weather <- Final_merged[, 3060:3075] # all numeric

y <- blues[, 4]
geno <- as.matrix(geno)
merged_rest <- data.frame(env, meta, weather) 

ETA <- list(
  list(X = geno, model = "BayesA"),   # BayesA for genotype data (G)
  list(X = merged_rest, model = "BRR")  # BRR for rest of the data
)

# Leave-one year out CV
set.seed(123)  # Make the CV reproducible
year <- Final_merged$Year
years_uniq <- unique(Final_merged$Year)

Correlation <- numeric(length(years_uniq))
RMSE <- numeric(length(years_uniq))

for (i in seq_along(years_uniq)) {
  year_test <- years_uniq[i]
  test_index <- which(year == year_test)
  
  y_train <- y
  y_train[test_index] <- NA 
  
  # Fit model
  model <- BGLR(y = y_train, ETA = ETA, nIter = 5000, burnIn = 500, thin = 5)
  
  actual <- y[test_index]
  predicted <- model$yHat[test_index]
  
  # Save metrics
  Correlation[i] <- cor(predicted, actual)
  RMSE[i] <- sqrt(mean((predicted - actual)^2))
  
  # Save actual vs predicted for each test year
  fold_df <- data.frame(Actual = actual, Predicted = predicted)
  write.csv(fold_df, paste0("fold_test_", year_test, ".csv"), row.names = FALSE)
}

# Pooled performance = mean across folds
correlation_pooled <- mean(Correlation, na.rm = TRUE)
rmse_pooled <- mean(RMSE, na.rm = TRUE)

# Combine results into a dataframe
cv_LOYO_results <- data.frame(
  Year = c(years_uniq, "Pooled"),
  Correlation = c(Correlation, correlation_pooled),
  RMSE = c(RMSE, rmse_pooled)
)

# Save to CSV
write.csv(cv_LOYO_results, 
          file = "Bayes_leave_one_year_out_cv_yield_results_scaling_removed.csv", 
          row.names = FALSE)
