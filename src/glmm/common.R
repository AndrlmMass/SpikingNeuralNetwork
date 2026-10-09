# =============================================================================
# Shared setup for the three revision-study GLMM scripts
# =============================================================================
# Sourced by fit_sweep.R, fit_baselines.R and fit_ablation_datasets.R, each of
# which is run on its own from the repo root:
#
#   Rscript src/glmm/fit_sweep.R
#   Rscript src/glmm/fit_baselines.R
#   Rscript src/glmm/fit_ablation_datasets.R
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

# `%||%` entered base R in 4.4 and glmmTMB's print methods use it; define it for
# older R versions so the scripts run there too.
if (!exists("%||%")) `%||%` <- function(a, b) if (is.null(a)) b else a

REPO <- getwd()
OUT <- file.path(REPO, "results", "glmm")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

cat("repo:", REPO, "\n")

# Beta regression needs y strictly inside (0, 1). Nothing in these runs is
# exactly 0 or 1, but guard anyway so a future all-correct cell cannot silently
# drop rows. N_TEST sets the size of that nudge (1 / (2 * N_TEST)). Every
# recorded test accuracy is a multiple of 1/200, i.e. computed on 200 images.
# No run is at 0 or 1, so the value does not change any fit.
N_TEST <- 200
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

# Dataset-specific predictions for the figures: fixed effects plus each dataset's
# random intercept, with seed (and dataset:seed) at zero via a seed level the
# model has not seen. 95% CIs on the logit scale, back-transformed.
cond_pred <- function(fit, newdata, seed) {
    newdata$seed <- factor("new", levels = c(levels(seed), "new"))
    p <- predict(fit, newdata = newdata, re.form = NULL, allow.new.levels = TRUE,
                 se.fit = TRUE)
    newdata$seed <- NULL
    newdata$fit <- plogis(p$fit)
    newdata$lo <- plogis(p$fit - qnorm(0.975) * p$se.fit)
    newdata$hi <- plogis(p$fit + qnorm(0.975) * p$se.fit)
    newdata
}

# Population-level predictions: fixed effects only, all random effects at zero
# (a typical dataset and seed). 95% confidence intervals for the mean, on the
# logit scale, back-transformed.
pop_pred <- function(fit, newdata) {
    p <- predict(fit, newdata = newdata, re.form = NA, se.fit = TRUE)
    newdata$fit <- plogis(p$fit)
    newdata$lo <- plogis(p$fit - qnorm(0.975) * p$se.fit)
    newdata$hi <- plogis(p$fit + qnorm(0.975) * p$se.fit)
    newdata
}

# Fixed-effect rows of the model summary for one factor: with treatment coding,
# the intercept is the reference level and each other coefficient is that level
# minus the reference (logit scale, Wald z tests).
coef_rows <- function(fit, factor_name) {
    s <- summary(fit)$coefficients$cond
    data.frame(level = sub(paste0("^", factor_name), "", rownames(s)),
               estimate = s[, "Estimate"], SE = s[, "Std. Error"],
               z.ratio = s[, "z value"], p.value = s[, "Pr(>|z|)"],
               row.names = NULL)
}

random_effects <- function(fit) {
    vcl <- glmmTMB::VarCorr(fit)$cond
    re <- data.frame(group = names(vcl),
                     variance = sapply(vcl, function(v) as.numeric(v)[1]),
                     row.names = NULL)
    re$sd <- sqrt(re$variance)
    re
}

# --- LaTeX helpers shared by the three tables ---------------------------------
apa <- function(x, d = 3) {
    x <- round(x, d) + 0                     # + 0 turns -0 into 0 (no "-.000")
    out <- sprintf(paste0("%.", d, "f"), x)
    out <- sub("^0\\.", ".", out)
    sub("^-0\\.", "-.", out)
}
apa_p <- function(p) ifelse(p < 0.001, "$<.001$", apa(p, 3))
small <- function(x, d = 3) ifelse(x < 10^(-d), sprintf("$< .%s1$", strrep("0", d - 1)), apa(x, d))

re_tex <- function(re) {
    pretty <- c(dataset = "Dataset", seed = "Seed", "dataset:seed" = "Dataset $\\times$ seed")
    out <- c("        \\midrule",
             "        \\multicolumn{2}{l}{Random effect variance}\\\\",
             "        Groups & Name & Variance & Std.Dev.& &\\\\")
    for (i in seq_len(nrow(re))) {
        g <- if (!is.na(pretty[re$group[i]])) pretty[[re$group[i]]] else re$group[i]
        out <- c(out, sprintf("        %s & (Intercept) & %s & %s & &\\\\",
                              g, small(re$variance[i]), small(re$sd[i])))
    }
    out
}

# --- Ablation runs, shared by fit_ablation_datasets.R and ablation_figure_data.R
# Reads both ablation summaries (MNIST; FMNIST, KMNIST, notMNIST) and returns
#   runs     the 16 factorial conditions, components as factors with "off" as
#            the reference level, accuracy squeezed into (0, 1) as `acc`
#   nosleep  the no-sleep reference runs (code "none"), which are not modelled
ABLATION_COMPONENTS <- c("downscale", "noise", "stdp", "suppress")
read_ablation <- function() {
    comps <- ABLATION_COMPONENTS
    read_one <- function(p) {
        d <- read.csv(p, colClasses = c(code = "character"))
        d[, c("code", "dataset", "seed", comps, "test_accuracy", "sleep_rate")]
    }
    ab <- rbind(read_one(file.path(REPO, "results", "ablation", "ablation_summary.csv")),
                read_one(file.path(REPO, "results", "ablation_datasets", "ablation_summary.csv")))
    stopifnot(length(unique(ab$sleep_rate[ab$code != "none"])) == 1)
    n_missing <- sum(is.na(ab$test_accuracy))
    if (n_missing) cat("WARNING:", n_missing, "runs without an accuracy are dropped\n")
    ab <- ab[!is.na(ab$test_accuracy), ]
    if (any(table(ab$code, ab$dataset) != 5))
        cat("WARNING: not every condition x dataset cell has 5 seeds\n")

    runs <- ab[ab$code != "none", ]
    for (v in comps) {
        on <- tolower(as.character(runs[[v]])) %in% c("true", "1")
        runs[[v]] <- factor(ifelse(on, "on", "off"), levels = c("off", "on"))
    }
    runs$acc <- squeeze(runs$test_accuracy, N_TEST)
    runs$dataset <- factor(runs$dataset, levels = c("mnist", "fmnist", "kmnist", "notmnist"))
    runs$seed <- factor(runs$seed)
    list(runs = runs, nosleep = ab[ab$code == "none", ])
}
