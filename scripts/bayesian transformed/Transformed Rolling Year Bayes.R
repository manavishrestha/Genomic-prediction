# dom rolling yield bayes

# Import Packages
library(BGLR)

# Import datasets
genotype_data <- read.csv("March19_Final_filtered_Recoded_imputed_SNP.csv")
Final_merged <- read.csv("Final_merged_March19.csv")
env_data <- read.csv("Env_clean.csv")
meta_data <- read.csv("metadata_clean_imputed.csv")
weather_data <- read.csv("weather_clean.csv")

# Data indices
blues <- Final_merged[,1:5]
geno <- Final_merged[,6:2402] # all numeric
env <- Final_merged[,2403:3056]
meta <- Final_merged[,3058:3059] # Exclude treatment
weather <- Final_merged[,3060:3075] # all numeric

# Specify target and features
y <- blues[,4]
geno <- as.matrix(geno)
merged_rest <- data.frame(env, meta, weather)

set.seed(123)

year <- Final_merged$Year
unique_years <- sort(unique(year))
yHat_result <- rep(NA, length(y))

# Total folds = one less than the number of years (since we predict year i+1)
folds <- length(unique_years) - 1

# Initialize cross-validation results
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
  
  cat("Train year:", train_years, "Test year:", test_year, "\n")
  
  geno_train <- geno[train_index, ]
  geno_test <- geno[test_index, ]
  
  # Check heterozygotes before
  het_before <- sum(geno == 1, na.rm = TRUE)
  
  # SNP mean calculation for transformation
  snp_mean = array(0, dim = c(ncol(geno_train), 3))
  for (snp in 1:ncol(geno_train)) {
    snp_mean[snp,1] = mean(y_train[geno_train[,snp] == 0], na.rm = TRUE)
    snp_mean[snp,2] = mean(y_train[geno_train[,snp] == 1], na.rm = TRUE)
    snp_mean[snp,3] = mean(y_train[geno_train[,snp] == 2], na.rm = TRUE)
  }
  snp_mean[is.na(snp_mean)] <- 0
  
  # Flip allele if needed
  snp_mean_temp = snp_mean
  to_flip = snp_mean_temp[,1] > snp_mean_temp[,3]
  snp_mean[to_flip,1] = snp_mean_temp[to_flip,3]
  snp_mean[to_flip,3] = snp_mean_temp[to_flip,1]
  
  geno_train[,to_flip] = abs(geno_train[,to_flip] - 2)
  geno_test[,to_flip] = abs(geno_test[,to_flip] - 2)
  
  # Dominance degree
  d = (snp_mean[,2] - snp_mean[,1]) / (snp_mean[,3] - snp_mean[,1]) * 2
  d[is.na(d)] = 1
  d[d >= 2] = 2
  d[d <= 0] = 0
  
  # Apply transformation to heterozygotes
  for (snp in 1:ncol(geno_train)) {
    het_train = which(geno_train[,snp] == 1)
    geno_train[het_train, snp] = d[snp]
    
    het_test = which(geno_test[,snp] == 1)
    geno_test[het_test, snp] = d[snp]
  }
  
  # Check heterozygotes after
  het_after <- sum(geno_train == 1, na.rm = TRUE) + sum(geno_test == 1, na.rm = TRUE)
  cat("Heterozygotes coded as 1 before:", het_before, "| after:", het_after, "\n")
  
  # Merge train and test back into full matrix
  geno_full <- geno
  geno_full[train_index, ] <- geno_train
  geno_full[test_index, ] <- geno_test
  
  ETA <- list(
    list(X = geno_full, model = "BayesA"),
    list(X = merged_rest, model = "BRR")
  )
  
  model <- BGLR(y = y_train, ETA = ETA, nIter = 5000, burnIn = 500, thin = 5)
  
  yHat_result[test_index] <- model$yHat[test_index]
  
  correlation_cv[i] <- cor(model$yHat[test_index], y[test_index])
  rmse_cv[i] <- sqrt(mean((model$yHat[test_index] - y[test_index])^2))
  
  # Save per-fold Actual vs Predicted
  fold_df <- data.frame(
    Actual = y[test_index],
    Predicted = model$yHat[test_index],
    Year = test_year
  )
  
  # Save as CSV
  write.csv(fold_df, file = paste0("dom_yieldRolling_Year_Fold_", test_year, "_Results.csv"), row.names = FALSE)
  
  # Save as RDS
  saveRDS(fold_df, file = paste0("dom_yield_Rolling_Year_Fold_", test_year, "_Results.rds"))
}

# Pooled performance = mean across folds
correlation_pooled <- mean(correlation_cv, na.rm = TRUE)
rmse_pooled <- mean(rmse_cv, na.rm = TRUE)

cv_summary <- data.frame(
  correlation = c(correlation_cv, Pooled = correlation_pooled),
  RMSE = c(rmse_cv, Pooled = rmse_pooled)
)

write.csv(cv_summary, "dom_yield_Rolling_Year_Yield.csv", row.names = TRUE)
