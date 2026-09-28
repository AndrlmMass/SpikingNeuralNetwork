# =============================================================================
# BASELINES — sleep against conventional stabilization (R3.1)
# =============================================================================
#   accuracy ~ regularization method (5 levels)
#
# Dataset is a RANDOM intercept here, unlike in fit_sweep.R where it is a fixed
# effect crossed with sleep duration. The difference is deliberate: in the sweep
# the per-dataset profile IS the finding, so dataset has to be in the fixed part;
# here dataset is only a nuisance source of level shifts, the question is the
# method ranking, and no method x dataset claim is made. Anything reported from
# this fit is therefore a method effect for a typical dataset.
#
# Usage (from the repo root):
#   Rscript src/glmm/fit_baselines.R
#
# Inputs:  results/baselines/baselines_summary.csv
# Outputs: results/glmm/baselines_coefs.csv
#          results/glmm/baselines_coefs_ref_sleep.csv
#          results/glmm/baselines_predictions.csv
# =============================================================================

source("src/glmm/common.R")
suppressPackageStartupMessages(library(emmeans))

bl <- read.csv(file.path(REPO, "results", "baselines", "baselines_summary.csv"))
bl <- bl[!is.na(bl$test_accuracy), ]
bl$acc <- squeeze(bl$test_accuracy, 700)
# `none` as reference so each coefficient reads as that method's gain over
# unregularized STDP — the contrast the reviewer asked about.
bl$method <- relevel(factor(bl$method), ref = "none")
bl$dataset <- factor(bl$dataset); bl$seed <- factor(bl$seed)

fit_bl <- glmmTMB(
    acc ~ method + (1 | dataset) + (1 | seed),
    family = beta_family(link = "logit"), data = bl
)
report(fit_bl, "BASELINES  acc ~ method + (1|dataset) + (1|seed)")

write.csv(tidy_fit(fit_bl, "baselines"), file.path(OUT, "baselines_coefs.csv"), row.names = FALSE)

# Predicted accuracy per method, for a typical dataset and seed (random effects
# at zero). emmeans back-transforms the interval endpoints from the logit scale.
emm <- emmeans(fit_bl, ~ method)
s <- summary(emm, type = "response", infer = c(TRUE, FALSE))
write.csv(data.frame(method = as.character(s$method), fit = s$response,
                     lo = s$asymp.LCL, hi = s$asymp.UCL),
          file.path(OUT, "baselines_predictions.csv"), row.names = FALSE)

# The contrast the paper actually has to report: sleep vs each conventional
# method. This is a contrast of the fit above, not a second model -- refitting
# with a different reference level would give the same numbers the long way.
ctr_sleep <- contrast(emm, "trt.vs.ctrl",
                      ref = which(levels(bl$method) == "sleep"), adjust = "none")
cat("\n--- sleep as reference (each method minus sleep) ---\n")
print(ctr_sleep)
cs <- as.data.frame(ctr_sleep)
write.csv(data.frame(study = "baselines_ref_sleep", term = as.character(cs$contrast),
                     estimate = cs$estimate, std_error = cs$SE,
                     z = cs$z.ratio, p = cs$p.value),
          file.path(OUT, "baselines_coefs_ref_sleep.csv"), row.names = FALSE)

cat("\nwrote baselines_coefs.csv, baselines_coefs_ref_sleep.csv,",
    "baselines_predictions.csv\n")
