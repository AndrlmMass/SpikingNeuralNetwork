# =============================================================================
# GLMM Comparison: Optimal Sleep vs Normalization vs No-Reg
# =============================================================================
# Two separate GLMMs:
#   Dataset A: accuracy-optimal sleep config vs normalization vs no-reg
#   Dataset B: clustering-optimal sleep config vs normalization vs no-reg
#
# Main model (model_acc / model_phi):
#   2-way interaction: reg_type * reg_mode (sleep vs normalize, no-reg excluded)
#
# No-reg contrast model (model_nreg_acc / model_nreg_phi):
#   Main effect only: reg_type (all three levels, no interaction)
#
# Usage: Rscript experiments/noise_article/GLMM/glmm_comparison.r
# =============================================================================

library(dplyr)
library(readxl)
library(glmmTMB)
library(writexl)

RESULTS_DIR <- file.path("..", "..", "..", "..", "SpikingNeuralNetwork", "results", "acc_history", "mnist", "2026.05.24", "21")

# ---------------------------------------------------------------------------
# Helper: Smithson-Verkuilen squish for beta family (keeps values off 0 and 1)
# ---------------------------------------------------------------------------
squish <- function(x, n) (x * (n - 1) + 0.5) / n

# ---------------------------------------------------------------------------
# Helper: fit + summarise + export predictions
# ---------------------------------------------------------------------------
fit_and_export <- function(df_full, outcome, reg_type_label,
                           family, link_fn, inv_link_fn,
                           out_pred_path, out_nreg_path) {
  cat(sprintf("\n===== %s | outcome: %s =====\n", reg_type_label, outcome))

  n <- nrow(df_full)

  # Squish outcome for beta family
  if (identical(family, beta_family())) {
    df_full[[paste0(outcome, "_t")]] <- squish(df_full[[outcome]], n)
    outcome_t <- paste0(outcome, "_t")
  } else {
    outcome_t <- outcome
  }

  # Reference levels
  df_full$reg_type <- factor(df_full$reg_type,
    levels = c("normalize", reg_type_label, "none")
  )
  df_full$reg_mode <- factor(df_full$reg_mode,
    levels = c("static", "layer", "neuron")
  )

  # ---- Main model: sleep vs normalize (2-way interaction) ----------------
  df_main <- df_full[df_full$reg_type != "none", ]
  df_main$reg_type <- droplevels(df_main$reg_type)

  formula_main <- as.formula(
    sprintf("%s ~ reg_type + (1|reg_mode) + (1 | seed)", outcome_t)
  )
  model_main <- glmmTMB(formula_main, data = df_main, family = family)
  cat("\n--- Main model (sleep vs normalize) ---\n")
  print(summary(model_main))

  # ---- No-reg contrast: all three reg_type levels, no interaction --------
  formula_nreg <- as.formula(
    sprintf("%s ~ reg_type + (1 | seed)", outcome_t)
  )
  model_nreg <- glmmTMB(formula_nreg, data = df_full, family = family)
  cat("\n--- No-reg contrast model ---\n")
  print(summary(model_nreg))

  # ---- Predictions from main model (2x3 factorial) -----------------------
  pred_grid <- expand.grid(
    reg_type = levels(df_main$reg_type),
    reg_mode = levels(df_main$reg_mode)
  )

  pred_link <- predict(model_main,
    newdata = pred_grid,
    type = "link", se.fit = TRUE, re.form = NA
  )
  pred_grid$fit <- inv_link_fn(pred_link$fit)
  pred_grid$lwr <- inv_link_fn(pred_link$fit - 1.96 * pred_link$se.fit)
  pred_grid$upr <- inv_link_fn(pred_link$fit + 1.96 * pred_link$se.fit)

  write_xlsx(pred_grid, out_pred_path)
  cat(sprintf("Saved predictions -> %s\n", out_pred_path))

  # ---- Predictions from no-reg model (3 reg_type levels) -----------------
  nreg_grid <- data.frame(
    reg_type = levels(df_full$reg_type),
    reg_mode = "layer" # reg_mode not in model; dummy value
  )
  # Model has no reg_mode term, so only reg_type matters
  nreg_link <- predict(model_nreg,
    newdata = nreg_grid,
    type = "link", se.fit = TRUE, re.form = NA
  )
  nreg_grid$fit <- inv_link_fn(nreg_link$fit)
  nreg_grid$lwr <- inv_link_fn(nreg_link$fit - 1.96 * nreg_link$se.fit)
  nreg_grid$upr <- inv_link_fn(nreg_link$fit + 1.96 * nreg_link$se.fit)
  nreg_grid$reg_mode <- NULL

  write_xlsx(nreg_grid, out_nreg_path)
  cat(sprintf("Saved no-reg contrast -> %s\n", out_nreg_path))

  list(
    model_main = model_main, model_nreg = model_nreg,
    pred = pred_grid, nreg_pred = nreg_grid
  )
}

# ---------------------------------------------------------------------------
# Dataset A: accuracy-optimal sleep
# ---------------------------------------------------------------------------
df_acc <- read_xlsx(file.path(RESULTS_DIR, "GLMM_comparison_acc.xlsx"))

results_acc <- fit_and_export(
  df_full          = df_acc,
  outcome          = "test_acc",
  reg_type_label   = "sleep_opt_acc",
  family           = beta_family(),
  link_fn          = qlogis,
  inv_link_fn      = plogis,
  out_pred_path    = file.path(RESULTS_DIR, "GLMM_comparison_acc_predictions.xlsx"),
  out_nreg_path    = file.path(RESULTS_DIR, "GLMM_comparison_acc_nreg.xlsx")
)

# ---------------------------------------------------------------------------
# Dataset B: clustering-optimal sleep
# ---------------------------------------------------------------------------
df_phi <- read_xlsx(file.path(RESULTS_DIR, "GLMM_comparison_phi.xlsx"))

# Note: static (sleep_dur=300, noise=100) is a pathological outlier for
# clustering — high noise collapses accuracy to ~12% while phi artificially
# inflates to ~58. Excluded from the phi main model; retained in no-reg
# contrast only for completeness.
df_phi_main <- df_phi[!(df_phi$reg_type == "sleep_opt_phi" &
  df_phi$reg_mode == "static"), ]

cat("\nNote: static sleep_opt_phi rows excluded from phi main model (outlier).\n")
cat(sprintf("Rows in phi main model dataset: %d\n", nrow(df_phi_main)))

results_phi <- fit_and_export(
  df_full          = df_phi_main,
  outcome          = "test_phi",
  reg_type_label   = "sleep_opt_phi",
  family           = Gamma(link = "log"),
  link_fn          = log,
  inv_link_fn      = exp,
  out_pred_path    = file.path(RESULTS_DIR, "GLMM_comparison_phi_predictions.xlsx"),
  out_nreg_path    = file.path(RESULTS_DIR, "GLMM_comparison_phi_nreg.xlsx")
)

cat("\nDone.\n")

