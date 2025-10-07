# Transformed yield Leave one year out Bayesian

library(BGLR)

#import datasets
genotype_data <- read.csv("March19_Final_filtered_Recoded_imputed_SNP.csv")
Final_merged <- read.csv("Final_merged_March19.csv")
env_data <- read.csv("Env_clean.csv")
meta_data <- read.csv("metadata_clean_imputed.csv")
weather_data <- read.csv("weather_clean.csv")

#as separate dataframes for separate matrix
blues <- Final_merged[ ,1:5]
geno <- Final_merged[ ,6:2402] #all num
env  <- Final_merged[ ,2403: 3056]
meta <- Final_merged[ , 3058:3059] #Exclude treatment
weather <- Final_merged[  ,3060:3075] #all num
rest <- Final_merged[ ,2403:3075]
y <- blues [ ,4]
geno <- as.matrix(geno)

merged_rest <- data.frame(env,meta,weather) 

#Leave-one year out cv
save_dir <- "/Users/manavi/Desktop/Final datasets and Results/z"
set.seed(123)  
year <- Final_merged$Year
years_uniq <- unique(Final_merged$Year)
yHat <- rep(NA, length(y))  
Correlation<- numeric(length(years_uniq))
RMSE <- numeric(length(years_uniq))

for (i in seq_along(years_uniq)) {
  year_test <- years_uniq[i]
  test_index <- which(year == year_test)
  train_index <- setdiff(1:length(y), test_index)
  
  y_train <- y
  y_train[test_index] <- NA 
  
  geno_train <- geno[train_index, ]
  geno_test <- geno[test_index, ]
  
  #Snp mean
  snp_mean = array(0, dim = c(ncol(geno_train), 3)) #an array with snp 0, 1 ,2 as columns for each snp
  for(snp in 1:ncol(geno_train)) {
    snp_mean[snp,1] = mean(y_train[geno_train[,snp]==0], na.rm = TRUE)
    snp_mean[snp,2] = mean(y_train[geno_train[,snp]==1], na.rm = TRUE)
    snp_mean[snp,3] = mean(y_train[geno_train[,snp]==2], na.rm = TRUE)
  }
  snp_mean[is.na(snp_mean)] <- 0
  
  #If mean[0] > mean[2]; flip the allele mean values
  snp_mean_temp = snp_mean
  to_flip = snp_mean_temp[,1] > snp_mean_temp[,3]
  snp_mean[to_flip,1] = snp_mean_temp[to_flip,3]
  snp_mean[to_flip,3] = snp_mean_temp[to_flip,1]
  
  geno_train[,to_flip] = abs(geno_train[,to_flip] - 2)
  geno_test[,to_flip] = abs(geno_test[,to_flip] - 2)
  
  #Calculate degree of dominance
  d = (snp_mean[,2] - snp_mean[,1]) / (snp_mean[,3] - snp_mean[,1]) * 2
  d[is.na(d)] = 1
  d[d >= 2] = 2    
  d[d <= 0] = 0    
  
  # Store pre-transformation heterozygote count
  pre_hets_train <- sum(geno_train == 1, na.rm = TRUE)
  
  # Add the weights (dominance transformation)
  for (snp in 1:ncol(geno)) {
    het_train_idx <- which(geno_train[, snp] == 1)
    het_test_idx <- which(geno_test[, snp] == 1)
    geno_train[het_train_idx, snp] <- d[snp]
    geno_test[het_test_idx, snp] <- d[snp]
  }
  
  # Check transformation of heterozygotes
  post_hets_train <- sum(geno_train == 1, na.rm = TRUE)
  cat("Heterozygotes coded as 1 before:", pre_hets_train, "| after:", post_hets_train, "\n")
  
  cat("Sample of transformed SNP values:\n")
  print(head(geno_train[, sample(1:ncol(geno_train), 5)]))
  
  # Merge train and test into full matrix
  geno_full <- geno
  geno_full[train_index, ] <- geno_train
  geno_full[test_index, ] <- geno_test
  
  ETA <- list(
    list(X = geno_full, model = "BayesA"),
    list(X = merged_rest, model = "BRR")
  )
  
  # Fit model - Example with BGLR
  model <- BGLR(y = y_train, ETA = ETA, nIter = 5000, burnIn = 500, thin = 5)
  
  yHat[test_index] <- model$yHat[test_index]
  actual <- y[test_index]
  predicted <- model$yHat[test_index]
  Correlation[i] <- cor(model$yHat[test_index], y[test_index])
  RMSE[i] <- sqrt(mean((model$yHat[test_index] - y[test_index])^2))
  
  ## Save actual vs predicted
  fold_df <- data.frame(Actual = actual, Predicted = predicted)
  write.csv(fold_df, paste0("dom_fold_test_bayes_loyo", year_test, ".csv"), row.names = FALSE)
  
  #Save rds
  saveRDS(model, file = paste0(save_dir, "/dom_fm_fold_yield_bayes_LOYO", year_test, ".rds"))
}

# Pooled performance as mean across folds
pooled_correlation <- mean(Correlation)
pooled_rmse <- mean(RMSE)

# Combine results into a dataframe
cv_LOYO_results <- data.frame(
  Year = c(years_uniq, "Pooled"),
  Correlation = c(Correlation, pooled_correlation),
  RMSE = c(RMSE, pooled_rmse)
)

# Save to CSV
write.csv(cv_LOYO_results, file = "dom_Bayes_leave_one_year_out_cv_yield_results_scaling_removed.csv", row.names = FALSE)
