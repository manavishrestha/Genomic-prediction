# Dom LOEO Yield Bayes 

library(BGLR)
library(data.table)
library(dplyr)

set.seed(42)

# Read data
genotype_data <- read.csv("March19_Final_filtered_Recoded_imputed_SNP.csv")
Final_merged <- read.csv("Final_merged_March19.csv")
env_data <- read.csv("Env_clean.csv")
meta_data <- read.csv("metadata_clean_imputed.csv")
weather_data <- read.csv("weather_clean.csv")

# Data partitions
blues <- Final_merged[,1:5]
geno <- as.matrix(Final_merged[,6:2402])
env <- Final_merged[,2403:3056]
meta <- Final_merged[,3058:3059]
weather <- Final_merged[,3060:3075]

# Target
y <- blues[,4]  # 'Average_BLUE'
merged_rest <- data.frame(env, meta, weather)

# Field locations and groups
field_locations <- unique(Final_merged$Field_Location)
n_groups <- 5
num_iterations <- 5

assign_locations_to_groups <- function(locations, n_groups) {
  shuffled <- sample(locations)
  split_list <- split(shuffled, cut(seq_along(shuffled), n_groups, labels = FALSE))
  location_to_group <- unlist(lapply(seq_along(split_list), function(i) 
    setNames(rep(paste0("Group_", i), length(split_list[[i]])), split_list[[i]])
  ))
  return(location_to_group)
}

all_results <- list()

for (iteration in 1:num_iterations) {
  cat(sprintf("Iteration %d:\n", iteration))
  location_to_group <- assign_locations_to_groups(field_locations, n_groups)
  Final_merged$Group <- location_to_group[Final_merged$Field_Location]
  
  # Save location assignments
  group_assignments <- split(names(location_to_group), location_to_group)
  assign_file <- sprintf("dom_iteration_%d_group_assignments.txt", iteration)
  writeLines(unlist(lapply(names(group_assignments), function(group) {
    paste(group, ":", paste(group_assignments[[group]], collapse = ", "))
  })), assign_file)
  
  groups <- unique(Final_merged$Group)
  
  mse_scores <- c()
  rmse_scores <- c()
  correlation_scores <- c()
  
  for (fold in 1:n_groups) {
    test_group <- groups[fold]
    train_groups <- setdiff(groups, test_group)
    
    train_index <- Final_merged$Group %in% train_groups
    test_index <- Final_merged$Group == test_group
    
    y_train <- y
    y_train[test_index] <- NA
    
    cat("Train Group(s):", paste(train_groups, collapse = ", "), "| Test Group:", test_group, "\n")
    
    geno_train <- geno[train_index, ]
    geno_test <- geno[test_index, ]
    
    # Check heterozygotes before
    het_before <- sum(geno == 1, na.rm = TRUE)
    
    # Dominance transformation
    snp_mean = array(0, dim = c(ncol(geno_train), 3))
    for (snp in 1:ncol(geno_train)) {
      snp_mean[snp,1] = mean(y_train[geno_train[,snp] == 0], na.rm = TRUE)
      snp_mean[snp,2] = mean(y_train[geno_train[,snp] == 1], na.rm = TRUE)
      snp_mean[snp,3] = mean(y_train[geno_train[,snp] == 2], na.rm = TRUE)
    }
    snp_mean[is.na(snp_mean)] <- 0
    
    snp_mean_temp = snp_mean
    to_flip = snp_mean_temp[,1] > snp_mean_temp[,3]
    snp_mean[to_flip,1] = snp_mean_temp[to_flip,3]
    snp_mean[to_flip,3] = snp_mean_temp[to_flip,1]
    
    geno_train[,to_flip] = abs(geno_train[,to_flip] - 2)
    geno_test[,to_flip] = abs(geno_test[,to_flip] - 2)
    
    d = (snp_mean[,2] - snp_mean[,1]) / (snp_mean[,3] - snp_mean[,1]) * 2
    d[is.na(d)] = 1
    d[d >= 2] = 2
    d[d <= 0] = 0
    
    for (snp in 1:ncol(geno_train)) {
      het_train = which(geno_train[,snp] == 1)
      geno_train[het_train, snp] = d[snp]
      
      het_test = which(geno_test[,snp] == 1)
      geno_test[het_test, snp] = d[snp]
    }
    
    het_after <- sum(geno_train == 1, na.rm = TRUE) + sum(geno_test == 1, na.rm = TRUE)
    cat("Heterozygotes coded as 1 before:", het_before, "| after:", het_after, "\n")
    
    geno_full <- geno
    geno_full[train_index, ] <- geno_train
    geno_full[test_index, ] <- geno_test
    
    ETA <- list(
      list(X = geno_full, model = "BayesA"),
      list(X = merged_rest, model = "BRR")
    )
    
    model <- BGLR(y = y_train, ETA = ETA, nIter = 5000, burnIn = 500, verbose = FALSE)
    
    y_pred <- model$yHat[test_index]
    y_true <- y[test_index]
    
    # Metrics
    mse <- mean((y_true - y_pred)^2)
    rmse <- sqrt(mse)
    correlation <- cor(y_true, y_pred)
    
    mse_scores <- c(mse_scores, mse)
    rmse_scores <- c(rmse_scores, rmse)
    correlation_scores <- c(correlation_scores, correlation)
    
    # Save predictions
    pred_results <- data.frame(
      Iteration = iteration,
      Fold = fold,
      Actual = y_true,
      Predicted = y_pred
    )
    write.csv(pred_results, sprintf("dom_env_iter_%d_fold_%d_results.csv", iteration, fold), row.names = FALSE)
  }
  
  # Summary for iteration (pooled = mean across folds)
  pooled_mse <- mean(mse_scores)
  pooled_rmse <- mean(rmse_scores)
  pooled_correlation <- mean(correlation_scores)
  
  iteration_results <- data.frame(
    Fold = paste0("Fold_", 1:n_groups),
    MSE = mse_scores,
    RMSE = rmse_scores,
    Correlation = correlation_scores
  )
  
  iteration_results <- rbind(iteration_results,
                             data.frame(Fold = "Pooled",
                                        MSE = pooled_mse,
                                        RMSE = pooled_rmse,
                                        Correlation = pooled_correlation))
  
  all_results[[iteration]] <- iteration_results
  cat(sprintf("Iteration %d Summary: MSE=%.4f, RMSE=%.4f, Cor=%.4f\n%s\n",
              iteration, pooled_mse, pooled_rmse, pooled_correlation, strrep("=", 40)))
}

# Save final results
final_results <- bind_rows(all_results, .id = "Iteration")
write.csv(final_results, "dom_env_all_iterations_summary.csv", row.names = FALSE)
print(final_results)

# Extract only pooled rows
pooled_df <- final_results %>% 
  filter(Fold == "Pooled")
print(pooled_df)

# Save pooled
write.csv(pooled_df, "dom_pooled_entries.csv", row.names = FALSE)

# Calculate mean across iterations
mean_row <- pooled_df %>%
  summarise(
    Iteration = "joint",
    Fold = "mean",
    MSE = mean(MSE),
    RMSE = mean(RMSE),
    Correlation = mean(Correlation)
  )

# Append mean row to pooled_df
pooled_df <- bind_rows(pooled_df, mean_row)

# Save the updated dataframe
write.csv(pooled_df, "dom_mean_entries.csv", row.names = FALSE)
