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
  assign_file <- sprintf("iteration_%d_group_assignments.txt", iteration)
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
    
    train_idx <- Final_merged$Group %in% train_groups
    test_idx <- Final_merged$Group == test_group
    
    y_train <- y
    y_train[test_idx] <- NA  # Mask test observations
    
    ETA <- list(
      list(X = geno, model = "BayesA"),
      list(X = merged_rest, model = "BRR")
    )
    
    model <- BGLR(y = y_train, ETA = ETA, nIter = 5000, burnIn = 500, verbose = FALSE)
    
    # Save the model
    model_file <- sprintf("iteration_%d_fold_%d_model.rds", iteration, fold)
    saveRDS(model, model_file)
    
    # Save predictions
    y_pred <- model$yHat[test_idx]
    y_true <- y[test_idx]
    
    pred_results <- data.frame(
      Iteration = iteration,
      Fold = fold,
      Actual = y_true,
      Predicted = y_pred
    )
    pred_file <- sprintf("iteration_%d_fold_%d_predictions.csv", iteration, fold)
    write.csv(pred_results, pred_file, row.names = FALSE)
    
    # Calculate metrics
    mse <- mean((y_true - y_pred)^2)
    rmse <- sqrt(mse)
    correlation <- cor(y_true, y_pred)
    
    mse_scores <- c(mse_scores, mse)
    rmse_scores <- c(rmse_scores, rmse)
    correlation_scores <- c(correlation_scores, correlation)
    
    cat(sprintf("  Fold %d:\n", fold))
    cat(sprintf("    MSE: %.4f\n    RMSE: %.4f\n    Correlation: %.4f\n",
                mse, rmse, correlation))
  }
  
  # Summarize this iteration
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

# Combine and save all iterations
final_results <- bind_rows(all_results, .id = "Iteration")
write.csv(final_results, "Iteration_eval_results.csv", row.names = FALSE)

# Print final results
print(final_results)
