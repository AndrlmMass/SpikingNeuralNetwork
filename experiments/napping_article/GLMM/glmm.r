# =============================================================================
# GLMM Analysis — sleep regularization × noise × reg-target
# =============================================================================
# Model (supervisor-specified):
#   test_acc ~ regularize * (sleep_duration * var_noise + reg_target) + (1 | seed)
#
# Key design note on the "none" condition
# ----------------------------------------
# Rows where reg_mode == "none" have no sleep_duration or var_noise — those
# parameters simply don't exist. The interaction `regularize × sleep_duration`
# zeroes out these predictors for none rows regardless of their placeholder
# value, so none rows only contribute to the unregularised baseline intercept.
# We assign log_sleep = 0 (= log(1)) and log_noise = 0 (= log(1)) as neutral
# placeholders. For reg_target, none rows are assigned "static" (reference
# level); the interaction regularize:reg_target zeroes that out too.
#
# Usage:
#   Rscript experiments/noise_article/GLMM/glmm.r
#
# Input:  results/acc_history/mnist/2026.05.24/21/Results_phase2 EXT.xlsx
# =============================================================================

library(dplyr)
library(readxl)
library(glmmTMB)
library(writexl)

#setwd("C:/Users/Andreas/Documents/Github/SNN_paper_repo")

# --- load data ---------------------------------------------------------------
cat("Reading Results_phase2 EXT.xlsx ...\n")
data <- read_excel(
    "C:\\Users\\Andreas\\Documents\\Github\\SNN_paper_repo\\results/acc_history/mnist/2026.05.24/21/Results_phase2 EXT.xlsx"
)

cat(sprintf("Loaded %d rows, columns: %s\n", nrow(data),
            paste(colnames(data), collapse = ", ")))

# --- recode predictors -------------------------------------------------------
# regularize: binary indicator (Reg column already exists: 1 = sleep, 0 = none)
data$regularize <- as.integer(data$Reg)

# log-scaled continuous predictors; none rows → 0 (= log(1), neutral baseline)
data$log_sleep <- ifelse(
    data$reg_mode != "none",
    log(as.numeric(data$sleep_duration)),
    0
)
data$log_noise <- ifelse(
    data$reg_mode != "none",
    log(as.numeric(data$var_noise)),
    0
)

# reg_target: static / layer / neuron; "none" rows get reference level "static"
# (the interaction regularize:reg_target zeroes this out for none rows anyway)
data$reg_target <- factor(
    ifelse(data$reg_mode == "none", "static", data$reg_mode),
    levels = c("static", "layer", "neuron")
)

# seed as grouping factor
data$seed <- as.factor(data$seed)

# Smithson & Verkuilen (2006) transformation — beta family requires (0, 1) open
n <- nrow(data)
data$test_acc_t  <- (data$test_acc  * (n - 1) + 0.5) / n
data$test_phi_t  <- (data$test_phi  / 100 * (n - 1) + 0.5) / n   # phi is 0-100

cat(sprintf("\nRegularize counts: %s\n",
            paste(table(data$regularize), collapse = " / ")))
cat(sprintf("reg_target levels: %s\n",
            paste(levels(data$reg_target), collapse = ", ")))
cat(sprintf("log_sleep range:   [%.2f, %.2f]\n",
            min(data$log_sleep), max(data$log_sleep)))
cat(sprintf("log_noise range:   [%.2f, %.2f]\n\n",
            min(data$log_noise), max(data$log_noise)))

# --- fit models --------------------------------------------------------------
cat("Fitting model for test_acc ...\n")
model_acc <- glmmTMB(
    test_acc_t ~ regularize * (log_sleep * log_noise + reg_target) + (1 | seed),
    data   = data,
    family = beta_family()
)

cat("Fitting model for test_phi ...\n")
model_phi <- glmmTMB(
    test_phi_t ~ regularize * (log_sleep * log_noise + reg_target) + (1 | seed),
    data   = data,
    family = beta_family()
)

# --- summaries ---------------------------------------------------------------
cat("\n=== MODEL: test_acc ===\n")
print(summary(model_acc))

cat("\n=== MODEL: test_phi ===\n")
print(summary(model_phi))

# --- predictions on original scale -------------------------------------------
# Build a grid over the regularized condition for plotting
cat("\nGenerating predictions for regularized condition ...\n")

sleep_vals <- sort(unique(data$log_sleep[data$regularize == 1]))
noise_vals  <- sort(unique(data$log_noise[data$regularize == 1]))
modes       <- levels(data$reg_target)

pred_grid <- expand.grid(
    regularize = 1,
    log_sleep  = sleep_vals,
    log_noise  = noise_vals,
    reg_target = factor(modes, levels = levels(data$reg_target))
)

pred_grid$pred_acc <- predict(
    model_acc, newdata = pred_grid, re.form = NA, type = "response"
)
pred_grid$pred_phi <- predict(
    model_phi, newdata = pred_grid, re.form = NA, type = "response"
)

# Back-transform axes to original scale
pred_grid$sleep_duration <- round(exp(pred_grid$log_sleep))
pred_grid$var_noise      <- exp(pred_grid$log_noise)

# Also predict the unregularized baseline (one row, log_sleep=0, log_noise=0)
none_row <- data.frame(
    regularize = 0, log_sleep = 0, log_noise = 0,
    reg_target = factor("static", levels = levels(data$reg_target))
)
none_row$pred_acc <- predict(model_acc, newdata = none_row, re.form = NA, type = "response")
none_row$pred_phi <- predict(model_phi, newdata = none_row, re.form = NA, type = "response")
cat(sprintf(
    "\nNone (unregularized) baseline — pred_acc: %.4f  pred_phi: %.4f\n",
    none_row$pred_acc, none_row$pred_phi
))

# --- export ------------------------------------------------------------------
out_path <- "results/acc_history/mnist/2026.05.24/21/GLMM_predictions.xlsx"
write_xlsx(
    list(
        predictions  = pred_grid,
        none_baseline = none_row
    ),
    path = out_path
)
cat(sprintf("\nPredictions saved to %s\n", out_path))
