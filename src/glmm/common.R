# =============================================================================
# Shared setup for the three revision-study GLMM scripts
# =============================================================================
# Sourced by fit_sweep.R, fit_baselines.R and fit_ablation.R, each of which is
# run on its own from the repo root:
#
#   Rscript src/glmm/fit_sweep.R
#   Rscript src/glmm/fit_baselines.R
#   Rscript src/glmm/fit_ablation.R
#
# Why Beta and not beta-binomial: the manuscript's existing analysis is a
# hierarchical Beta regression on test accuracy. The RF-article convention is
# beta-binomial on raw counts, which is the better model when test-set sizes
# differ, but switching family mid-revision would make the new tables
# incommensurable with the ones the reviewers already read. Test-set size is
# constant within a dataset here, so the two agree closely.
# =============================================================================

suppressPackageStartupMessages({
    library(glmmTMB)
})

# `%||%` entered base R in 4.4; glmmTMB's print methods use it. This R is older,
# so define it here rather than upgrading the toolchain mid-revision.
if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a

REPO <- getwd()
OUT <- file.path(REPO, "results", "glmm")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

cat("repo:", REPO, "\n")

# Beta regression needs y strictly inside (0, 1). Nothing in these runs is
# exactly 0 or 1, but guard anyway so a future all-correct cell cannot silently
# drop rows.
squeeze <- function(y, n) {
    eps <- 1 / (2 * n)
    pmin(pmax(y, eps), 1 - eps)
}

tidy_fit <- function(fit, label) {
    s <- summary(fit)$coefficients$cond
    data.frame(
        study = label,
        term = rownames(s),
        estimate = s[, "Estimate"],
        std_error = s[, "Std. Error"],
        z = s[, "z value"],
        p = s[, "Pr(>|z|)"],
        row.names = NULL
    )
}

report <- function(fit, label) {
    cat("\n", strrep("=", 74), "\n", label, "\n", strrep("=", 74), "\n", sep = "")
    print(summary(fit))
    vc <- as.data.frame(glmmTMB::VarCorr(fit)$cond)
    if (nrow(vc)) {
        cat("\nrandom-effect SDs:\n")
        print(vc)
    }
    cat("\ndispersion (phi):", sigma(fit), "\n")
}
