# Rolling Year Cross Validation Yield


## Import Packages
library(BGLR)
library(caret)

## Import datasets
genotype_data <- read.csv("March19_Final_filtered_Recoded_imputed_SNP.csv")
Final_merged <- read.csv("Final_merged_March19.csv")
env_data <- read.csv("Env_clean.csv")
meta_data <- read.csv("metadata_clean_imputed.csv")
weather_data <- read.csv("weather_clean.csv")

## Data indices
blues <- Final_merged[,1:5]
geno <- Final_merged[,6:2402] # all numeric
env <- Final_merged[,2403:3056]
meta <- Final_merged[,3058:3059] # Exclude treatment
weather <- Final_merged[,3060:3075] # all numeric

## Specify target and features
y <- blues[,4]
geno <- as.matrix(geno)
merged_rest <- data.frame(env, meta, weather)

ETA <- list(
  list(X = geno, model = "BayesA"),   # BayesA for genotype data (G)
  list(X = merged_rest, model = "BRR")  # BRR for rest of the data
)

set.seed(123)

year <- Final_merged$Year
unique_years <- sort(unique(year))

folds <- length(unique_years) - 1

## cross-validation
correlation_cv <- rep(NA, times = folds) 
names(correlation_cv) <- paste('fold:', 1:folds, sep='')

rmse_cv <- rep(NA, times = folds)
names(rmse_cv) <- paste('fold:', 1:folds, sep='')

for (i in 1:folds) {
  train_years <- unique_years[1:i]
  test_year <- unique_years[i + 1]
  
  train_index <- which(year %in% train_years)
  test_index <- which(year == test_year)
  
  y_train <- y
  y_train[test_index] <- NA
  
  cat("Train year:", train_years,
      "Test year:", test_year, "\n")
  
  model <- BGLR(y = y_train, ETA = ETA, nIter = 5000, burnIn = 500, thin = 5)
  
  # Save correlation and RMSE for this fold
  correlation_cv[i] <- cor(model$yHat[test_index], y[test_index])
  rmse_cv[i] <- sqrt(mean((model$yHat[test_index] - y[test_index])^2))
}

## Pooled results = mean across folds
correlation_pooled <- mean(correlation_cv, na.rm = TRUE)
rmse_pooled <- mean(rmse_cv, na.rm = TRUE)

## Save results 
cv_summary <- data.frame(
  correlation = c(correlation_cv, Pooled = correlation_pooled),
  RMSE = c(rmse_cv, Pooled = rmse_pooled)
)

write.csv(cv_summary, "Rolling_Year_CV_Summary.csv", row.names = TRUE)
