# =============================================================================
# ABLATION — 2^4 component factorial (R3.2), MNIST only
# =============================================================================
#   accuracy ~ downscale * noise * stdp * suppress
#
# Usage (from the repo root):
#   Rscript src/glmm/fit_ablation.R
#
# Inputs:  results/ablation/ablation_summary.csv
# Outputs: results/glmm/ablation_coefs.csv
#          results/glmm/ablation_predictions.csv
#          results/glmm/ablation_nosleep_reference.csv
# =============================================================================

source("src/glmm/common.R")
suppressPackageStartupMessages(library(emmeans))

ab <- read.csv(file.path(REPO, "results", "ablation", "ablation_summary.csv"))
ab <- ab[!is.na(ab$test_accuracy), ]
ab_ref <- ab[ab$code == "none", ]          # no-sleep reference, not in the factorial
ab <- ab[ab$code != "none", ]
ab$acc <- squeeze(ab$test_accuracy, 700)
for (v in c("downscale", "noise", "stdp", "suppress")) {
    ab[[v]] <- factor(ifelse(tolower(as.character(ab[[v]])) %in% c("true", "1"),
                             "on", "off"), levels = c("off", "on"))
}
ab$seed <- factor(ab$seed)

fit_ab <- glmmTMB(
    acc ~ downscale * noise * stdp * suppress + (1 | seed),
    family = beta_family(link = "logit"), data = ab
)
report(fit_ab, "ABLATION  acc ~ downscale * noise * stdp * suppress + (1|seed)")
write.csv(tidy_fit(fit_ab, "ablation_full"),
          file.path(OUT, "ablation_coefs.csv"), row.names = FALSE)

# Note for interpreting the coefficients: conditions without downscaling
# collapse to a degenerate state, where three distinct configurations returned
# an identical multiset of accuracies across seeds -- the signature of weights
# saturating against the clip bounds rather than of a graded effect. Main
# effects are therefore pulled towards that floor.

# Per-condition predictions for the figure, over all 16 cells. These are what
# the figure's confidence intervals are drawn from. Random effects sit at zero,
# i.e. a typical seed; emmeans back-transforms the interval endpoints from the
# logit scale.
s <- summary(emmeans(fit_ab, ~ downscale * noise * stdp * suppress),
             type = "response", infer = c(TRUE, FALSE))
ab_pred <- data.frame(
    downscale = as.character(s$downscale), noise = as.character(s$noise),
    stdp = as.character(s$stdp), suppress = as.character(s$suppress),
    fit = s$response, lo = s$asymp.LCL, hi = s$asymp.UCL
)
ab_pred$code <- apply(ab_pred[, c("downscale", "noise", "stdp", "suppress")], 1,
                      function(r) paste0(ifelse(r == "on", "1", "0"), collapse = ""))
write.csv(ab_pred, file.path(OUT, "ablation_predictions.csv"), row.names = FALSE)

cat("\nno-sleep reference (not in the factorial): mean acc",
    mean(ab_ref$test_accuracy), "over", nrow(ab_ref), "seeds\n")
write.csv(data.frame(mean_acc = mean(ab_ref$test_accuracy), n = nrow(ab_ref)),
          file.path(OUT, "ablation_nosleep_reference.csv"), row.names = FALSE)

cat("\nwrote ablation_coefs.csv, ablation_predictions.csv,",
    "ablation_nosleep_reference.csv\n")
