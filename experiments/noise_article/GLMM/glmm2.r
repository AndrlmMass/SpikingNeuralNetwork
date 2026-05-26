# =============================================================================
# GLMM Analysis for SNN-sleepy vs snntorch comparison
# =============================================================================
# This script fits a Generalized Linear Mixed Model (GLMM) with beta family
# to compare accuracy across sleep durations and models for the MNIST family.
#
# Usage: Run from the GLM/ directory:
#   Rscript analysis.r
#
# Input:  Results_.xlsx (in current directory, GLM/)
# Output: pred.xlsx (in current directory, GLM/)
# =============================================================================

# Load libraries
library(dplyr)
library(tidyr)
library(readxl)
library(glmmTMB)
library(ggeffects)
library(ggplot2)
library(writexl)

# Read data from current directory (GLM/)
cat("Reading Results_.xlsx...\n")
data <- read_excel("C:\\Users\\Andreas\\Documents\\Github\\SNN_paper_repo\\results\\acc_history\\mnist\\2026.05.24\\21\\Results_phase2.xlsx")
log_data <- data
full_data <- read_excel("C:\\Users\\Andreas\\Documents\\Github\\SNN_paper_repo\\results\\acc_history\\mnist\\2026.05.24\\21\\Results_phase2 EXT.xlsx")

#convert logdata set to log-scale
log_data$sleep_duration <- log(log_data$sleep_duration)
log_data$var_noise <- log(log_data$var_noise)
full_data$sleep_duration <- log(full_data$sleep_duration)
full_data$var_noise <- log(full_data$var_noise)

# Convert to factors
#data$sleep_duration <- as.factor(data$sleep_duration)
#data$var_noise <- as.factor(data$var_noise)
colnames(data)[1] <- "reg_target"
data$reg_target <- as.factor(data$reg_target)

# Same for log-data
#log_data$sleep_duration <- as.factor(log_data$sleep_duration)
#log_data$var_noise <- as.factor(log_data$var_noise)
colnames(log_data)[1] <- "reg_target"
log_data$reg_target <- as.factor(log_data$reg_target)

# same for full dataset
colnames(full_data)[1] <- "reg_target"
colnames(full_data)[7] <- "reg"
full_data$reg <- as.integer(full_data$reg)
full_data$reg_target <- as.factor(full_data$reg_target)

cat(sprintf("Noise level: %s\n", paste(unique(data$var_noise), collapse = ", ")))
cat(sprintf("Sleep durations: %s\n", paste(unique(data$sleep_duration), collapse = ", ")))
cat(sprintf("Reg target: %s\n", paste(unique(data$reg_target), collapse = ", ")))

# Fit GLMM with beta family (for accuracy bounded 0-1)
cat("\nFitting GLMM model...\n")

model1 <- glmmTMB(
  test_acc ~ sleep_duration * var_noise + reg_target + (1 | seed),
  data = data,
  family = beta_family()
)

model2 <- glmmTMB(
  test_acc ~ sleep_duration * var_noise + reg_target + (1 | seed),
  data = log_data,
  family = beta_family()
)

model3 <- glmmTMB(
  test_acc ~ reg * (sleep_duration * var_noise + reg_target) + (1 | seed),
  data=full_data,
  family=beta_family()
)

# Print model summary
cat("\n=== MODEL SUMMARY ACCURACY ===\n")
print(summary(model1))

cat("\n=== MODEL SUMMARY CLUSTERING ===\n")
print(summary(model2))

# ---- prediction grid (on log scale, matching what the model sees) ----------
sleep_orig  <- c(1, 10, 50, 100, 150, 200, 300)
noise_orig  <- c(1.0, 2.5, 5.0, 7.5, 10.0, 100.0)

pred_grid <- expand.grid(
  sleep_duration = log(sleep_orig),
  var_noise      = log(noise_orig),
  reg_target     = levels(log_data$reg_target)
)

# Population-level predictions (re.form = NA ignores the random seed effect)
pred_grid$fit <- predict(model2, newdata = pred_grid,
                         type = "response", re.form = NA,
                         allow.new.levels = TRUE)

# Back-transform for readable axes
pred_grid$sleep_orig <- exp(pred_grid$sleep_duration)
pred_grid$noise_orig <- exp(pred_grid$var_noise)

# ---- plot -------------------------------------------------------------------
ggplot(pred_grid, aes(x = factor(sleep_orig), y = fit,
                      colour = factor(noise_orig), group = factor(noise_orig))) +
  geom_line() +
  geom_point() +
  facet_wrap(~ reg_target) +
  scale_colour_viridis_d(name = "Noise variance") +
  labs(x = "Sleep duration (timesteps)",
       y = "Predicted accuracy",
       title = "GLMM predicted accuracy") +
  theme_bw()

ggsave("C:\\Users\\Andreas\\Documents\\Github\\SNN_paper_repo\\results\\acc_history\\mnist\\2026.05.24\\21\\glmm_predictions.pdf",
       width = 10, height = 4)

# ---- export to Excel --------------------------------------------------------
write_xlsx(
  pred_grid[, c("reg_target", "sleep_orig", "noise_orig", "fit")],
  "C:\\Users\\Andreas\\Documents\\Github\\SNN_paper_repo\\results\\acc_history\\mnist\\2026.05.24\\21\\GLMM_predictions.xlsx"
)
