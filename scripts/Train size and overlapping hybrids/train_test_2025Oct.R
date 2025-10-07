# Train size and overlapping hybrids

#Leave One Year Out
## Import datasets
genotype_data <- read.csv("March19_Final_filtered_Recoded_imputed_SNP.csv")
Final_merged <- read.csv("Final_merged_March19.csv")
env_data <- read.csv("Env_clean.csv")
meta_data <- read.csv("metadata_clean_imputed.csv")
weather_data <- read.csv("weather_clean.csv")

## Leave-one year out cv

set.seed(123)  # Make the CV reproducible
year <- Final_merged$Year
years_uniq <- unique(Final_merged$Year)

## Separate dataframes for Bayes A and BRR
blues <- Final_merged[ ,1:5]
geno <- Final_merged[ ,6:2402] #all num
env  <- Final_merged[ ,2403: 3056]
meta <- Final_merged[ , 3058:3059] #Exclude treatment
weather <- Final_merged[  ,3060:3075] #all num

## Separate Feature and Target
y <- blues [ ,4]
geno <- as.matrix(geno)
merged_rest <- data.frame(env,meta,weather) 


# Initialize summary storage
cv_summary <- data.frame()

# Create or clear the overlap text file
fileConn <- file("hybrid_overlap_lists_LOYO.txt")
writeLines("", fileConn)
close(fileConn)

for (i in seq_along(years_uniq)) {
  year_test <- years_uniq[i]
  test_index <- which(year == year_test)
  train_index <- setdiff(1:length(y), test_index)
  years_train <- setdiff(years_uniq, year_test)
  cat("Test year:", year_test, "\n")
  cat("Train years:", paste(years_train, collapse = ", "), "\n")
  
  ## Get hybrid info
  test_hybrids <- unique(Final_merged$Hybrid[test_index])
  train_hybrids <- unique(Final_merged$Hybrid[train_index])
  overlap <- intersect(train_hybrids, test_hybrids)
  
  # Print or log
  cat("Train size:", length(train_index), "\n")
  cat("Test size:", length(test_index), "\n")
  
  # Save summary row
  cv_summary <- rbind(cv_summary, data.frame(
    Year_left_out = year_test,
    Train_size = length(train_index),
    Test_size = length(test_index),
    Test_hybrids = length(test_hybrids),
    Overlapping_hybrids = length(overlap)
  ))
  
  # Append overlap info to text file
  overlap_entry <- paste0(
    "Year left out: ", year_test, "\n",
    "Overlapping hybrids (", length(overlap), "):\n",
    paste(sort(overlap), collapse = ", "), "\n\n"
  )
  cat(overlap_entry, file = "hybrid_overlap_lists_LOYO.txt", append = TRUE)
}
write.csv(cv_summary,"Bayes_LOYO_train_test.csv", row.names = FALSE)

# Rolling Year
year <- Final_merged$Year
hybrid <- Final_merged$Hybrid
unique_years <- sort(unique(year))

folds <- length(unique_years) - 1

# Create a data frame for summary
summary_df <- data.frame(
  Fold = integer(folds),
  Train_Years = character(folds),
  Test_Year = integer(folds),
  Train_Size = integer(folds),
  Test_Size = integer(folds),
  Test_Hybrids = integer(folds),    
  Overlapping_Hybrids = integer(folds),
  stringsAsFactors = FALSE
)

# Open a connection for the .txt file to write overlapping hybrids
txt_file <- file("Oct_2025_rolling_overlapping_hybrids_details.txt", open = "wt")

for (i in 1:folds) {
  train_years <- unique_years[1:i]
  test_year <- unique_years[i + 1]
  
  train_index <- which(year %in% train_years)
  test_index <- which(year == test_year)
  
  train_hybrids <- unique(hybrid[train_index])
  test_hybrids <- unique(hybrid[test_index])
  
  overlap_hybrids <- intersect(train_hybrids, test_hybrids)
  
  # Save to data frame
  summary_df[i, ] <- list(
    Fold = i,
    Train_Years = paste(train_years, collapse = ", "),
    Test_Year = test_year,
    Train_Size = length(train_index),
    Test_Size = length(test_index),
    Test_Hybrids = length(test_hybrids), 
    Overlapping_Hybrids = length(overlap_hybrids)
  )
  
  # Write to text file
  writeLines(sprintf("Fold %d - Test Year: %d", i, test_year), txt_file)
  writeLines(sprintf("Overlapping Hybrids (%d):", length(overlap_hybrids)), txt_file)
  writeLines(paste(overlap_hybrids, collapse = ", "), txt_file)
  writeLines("", txt_file)  # blank line between folds
}

close(txt_file)

# Save the summary dataframe as CSV
write.csv(summary_df, file = "Oct_2025_Rolling_train_test_summary.csv", row.names = FALSE)

