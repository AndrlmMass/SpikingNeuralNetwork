# =============================================================================
# GLMM Analysis for SNN-sleepy vs snntorch comparison
# =============================================================================
# This script fits a Generalized Linear Mixed Model (GLMM) with beta family
# to compare accuracy across sleep durations and models for the MNIST family.
#
# Usage: Run from the src/analysis/ directory:
#   Rscript mixed_model2.r
#
# Input:  Results_.xlsx (in current directory)
# Output: pred.xlsx (in current directory)
# =============================================================================

# Load libraries
library(dplyr)
library(tidyr)
library(readxl)
library(glmmTMB)
library(ggeffects)
library(ggplot2)
library(writexl)

# Read data from current directory (src/analysis/)
cat("Reading Results_.xlsx...\n")
data <- read_excel("Results_.xlsx")

# Select data from Run == 1
data <- data[data$Run == 1, ]
cat(sprintf("Loaded %d rows after filtering Run==1\n", nrow(data)))

# Convert dataset values from wide to long format
ldata <- data %>%
  pivot_longer(
    cols = ends_with("mnist"),
    names_to = "Dataset",
    values_to = "Accuracy",
  )

# Convert to factors
ldata$Sleep_duration <- as.factor(ldata$Sleep_duration)
ldata$Model <- as.factor(ldata$Model)
ldata$Dataset <- as.factor(ldata$Dataset)

cat(sprintf("Long format: %d observations\n", nrow(ldata)))
cat(sprintf("Models: %s\n", paste(unique(ldata$Model), collapse=", ")))
cat(sprintf("Datasets: %s\n", paste(unique(ldata$Dataset), collapse=", ")))
cat(sprintf("Sleep durations: %s\n", paste(unique(ldata$Sleep_duration), collapse=", ")))

# Fit GLMM with beta family (for accuracy bounded 0-1)
cat("\nFitting GLMM model...\n")
model <- glmmTMB(
  Accuracy ~ Sleep_duration * Model + (1|Dataset) + (1|Dataset:Seed),
  data = ldata,
  family = beta_family()
)

# Print model summary
cat("\n=== MODEL SUMMARY ===\n")
print(summary(model))

# Get predictions with confidence intervals
cat("\nCalculating predictions...\n")
pred <- ggpredict(
  model,
  terms = c("Sleep_duration", "Model", "Dataset"),
  type = "random",
  bias_correction = TRUE
)

pred_df <- as.data.frame(pred)
cat(sprintf("Generated %d predictions\n", nrow(pred_df)))

# Export predictions
write_xlsx(x = pred_df, path = "pred.xlsx")
cat("\nPredictions saved to pred.xlsx\n")

