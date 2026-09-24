# =============================================================================
# GLMMs for the three revision studies
# =============================================================================
# Beta regression via glmmTMB, matching the family used for Table 1 in the
# submitted manuscript so the revised tables are comparable in form.
#
# Why Beta and not beta-binomial: the manuscript's existing analysis is a
# hierarchical Beta regression on test accuracy. The RF-article convention is
# beta-binomial on raw counts, which is the better model when test-set sizes
# differ, but switching family mid-revision would make the new tables
# incommensurable with the ones the reviewers already read. Test-set size is
# constant within a dataset here, so the two agree closely.
#
#   1. sweep      accuracy ~ sleep ratio (11 levels, 4 datasets, 5 seeds)
#   2. baselines  accuracy ~ regularization method (5 levels)
#   3. ablation   accuracy ~ downscale * noise * stdp * suppress (MNIST only)
#
# Usage:
#   Rscript src/glmm/fit_glmms.R
#
# Inputs:  results/{sweep,baselines,ablation}/*_summary.csv
# Outputs: results/glmm/{sweep,baselines,ablation}_{coefs,predictions}.csv
#          plus a console summary of each fit
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

# --- helpers -----------------------------------------------------------------

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

# =============================================================================
# 1. SWEEP — accuracy as a function of sleep ratio
# =============================================================================
sw <- read.csv(file.path(REPO, "results", "sweep", "sweep_summary.csv"))
sw <- sw[!is.na(sw$test_accuracy), ]
sw$n_test <- 700   # fixed within dataset; used only for the (0,1) squeeze
sw$acc <- squeeze(sw$test_accuracy, sw$n_test)

# Sleep ratio as a FACTOR, not continuous: the manuscript reports one
# coefficient per level against 0%, and the response is non-monotonic, so a
# linear term would misrepresent it.
sw$ratio_f <- relevel(factor(sw$sleep_rate), ref = "0")
sw$dataset <- factor(sw$dataset)
sw$seed <- factor(sw$seed)

fit_sw <- glmmTMB(
    acc ~ ratio_f + (1 | dataset) + (1 | seed) + (1 | dataset:seed),
    family = beta_family(link = "logit"), data = sw
)
report(fit_sw, "1. SWEEP  acc ~ factor(sleep_ratio) + (1|dataset) + (1|seed) + (1|dataset:seed)")

# Population-level predictions per ratio (random effects at zero)
nd <- data.frame(ratio_f = levels(sw$ratio_f))
nd$dataset <- NA; nd$seed <- NA
pr <- predict(fit_sw, newdata = nd, type = "link", se.fit = TRUE,
              allow.new.levels = TRUE, re.form = NA)
inv <- function(x) 1 / (1 + exp(-x))
sweep_pred <- data.frame(
    sleep_rate = as.numeric(as.character(nd$ratio_f)),
    fit = inv(pr$fit),
    lo = inv(pr$fit - 1.96 * pr$se.fit),
    hi = inv(pr$fit + 1.96 * pr$se.fit)
)
sweep_pred <- sweep_pred[order(sweep_pred$sleep_rate), ]
write.csv(sweep_pred, file.path(OUT, "sweep_predictions.csv"), row.names = FALSE)
write.csv(tidy_fit(fit_sw, "sweep"), file.path(OUT, "sweep_coefs.csv"), row.names = FALSE)

# --- regression table for the main run --------------------------------------
# A conventional GLMM table: fixed effects on the link scale, then the random
# effect SDs, then model fit. No back-transformed accuracy column -- the
# coefficients are the estimates, and the fitted accuracies belong in the
# figure and its prediction file, not in a regression table.
stars <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**",
                     ifelse(p < 0.05, "*", ifelse(p < 0.1, ".", ""))))
fmt_p <- function(p) ifelse(p < 1e-4, "< 0.0001", sprintf("%.4f", p))

co <- summary(fit_sw)$coefficients$cond
lv <- rownames(co)
tab <- data.frame(
    term = ifelse(lv == "(Intercept)", "Intercept (0\\% sleep)",
                  sprintf("Sleep duration %s\\%%",
                          suppressWarnings(as.numeric(sub("^ratio_f", "", lv))) * 100)),
    estimate = co[, "Estimate"],
    std_error = co[, "Std. Error"],
    z = co[, "z value"],
    p = co[, "Pr(>|z|)"],
    row.names = NULL
)
tab$sig <- stars(tab$p)
ord <- order(ifelse(lv == "(Intercept)", -1,
                    suppressWarnings(as.numeric(sub("^ratio_f", "", lv)))))
tab <- tab[ord, ]

vcl <- glmmTMB::VarCorr(fit_sw)$cond
re <- data.frame(
    term = sprintf("SD (%s)", names(vcl)),
    sd = sapply(names(vcl), function(nm) sqrt(as.numeric(vcl[[nm]])[1])),
    row.names = NULL
)

write.csv(tab[, c("term", "estimate", "std_error", "z", "p", "sig")],
          file.path(OUT, "sweep_table.csv"), row.names = FALSE)
write.csv(re, file.path(OUT, "sweep_table_random.csv"), row.names = FALSE)

# --- console ---
bar <- strrep("=", 68)
cat("\n", bar, "\n",
    "MAIN RUN -- Beta GLMM (logit link)\n",
    "acc ~ factor(sleep duration) + (1|dataset) + (1|seed) + (1|dataset:seed)\n",
    bar, "\n", sep = "")
cat(sprintf("%-26s %9s %8s %7s %11s\n", "Fixed effects", "Estimate", "SE", "z", "p"))
cat(strrep("-", 68), "\n", sep = "")
plain <- function(x) gsub("\\\\", "", x)
for (i in seq_len(nrow(tab))) {
    cat(sprintf("%-26s %9.4f %8.4f %7.2f %8s %-3s\n",
                plain(tab$term[i]), tab$estimate[i], tab$std_error[i],
                tab$z[i], fmt_p(tab$p[i]), tab$sig[i]))
}
cat(strrep("-", 68), "\n", sep = "")
cat(sprintf("%-26s %9s\n", "Random effects", "SD"))
for (i in seq_len(nrow(re))) {
    cat(sprintf("%-26s %9.4f\n", re$term[i], re$sd[i]))
}
cat(strrep("-", 68), "\n", sep = "")
cat(sprintf("%-26s %9.2f\n", "Dispersion (phi)", sigma(fit_sw)))
cat(sprintf("%-26s %9d\n", "Observations", nrow(sw)))
cat(sprintf("%-26s %9d\n", "Datasets", nlevels(sw$dataset)))
cat(sprintf("%-26s %9d\n", "Seeds", nlevels(sw$seed)))
cat(sprintf("%-26s %9.1f\n", "AIC", AIC(fit_sw)))
cat(bar, "\n", sep = "")
cat("Signif.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1\n")
cat("Reference level is 0 pct sleep; estimates are log-odds differences from it.\n")

# --- LaTeX, matching the manuscript's existing beta-regression table --------
# longtable with a repeated header, beta_{j=..} row labels, a separate
# significance column, and APA number style (no leading zero). The submitted
# version of this table also carried the SG-SNN main effect and the
# sleep x model interaction; that model is dropped from the revision, so those
# blocks are gone and only the sleep-duration effect remains.
apa <- function(x, d = 3) {
    out <- sprintf(paste0("%.", d, "f"), x)
    out <- sub("^0\\.", ".", out)
    sub("^-0\\.", "-.", out)
}
apa_p <- function(p) ifelse(p < 0.001, "$<.001$", apa(p, 3))

lab_beta <- function(term) {
    if (term == "(Intercept)") return("Intercept")
    sprintf("$\\beta_{j=%g}$",
            as.numeric(sub("^ratio_f", "", term)) * 100)
}

tex <- c(
    "{",
    "\\setlength{\\tabcolsep}{15pt} % Only affects this table",
    "\\begin{longtable}{lrrrrl}",
    "    \\caption{Mixed effect model with beta regression \\label{tab:betareg}}\\\\",
    "    \\toprule",
    "    Effect  & Estimate & Std. Error & \\textit{z} value & \\textit{p}-value & \\\\",
    "    \\endfirsthead",
    "",
    "    \\toprule",
    "    Effect  & Estimate & Std. Error & \\textit{z} value & \\textit{p}-value & \\\\",
    "    \\endhead"
)

# Intercept row, then the sleep-duration block.
ic <- which(rownames(co) == "(Intercept)")
tex <- c(tex, sprintf("        Intercept & %s & %s & %s & %s \\\\[6pt]",
                      apa(co[ic, "Estimate"]), apa(co[ic, "Std. Error"]),
                      apa(co[ic, "z value"]), apa_p(co[ic, "Pr(>|z|)"])))
tex <- c(tex,
    "        \\multicolumn{6}{l}{\\textit{Main effect of sleep duration}}\\\\")

rl <- rownames(co)
sleep_rows <- rl[rl != "(Intercept)"]
sleep_rows <- sleep_rows[order(as.numeric(sub("^ratio_f", "", sleep_rows)))]
for (term in sleep_rows) {
    i <- which(rl == term)
    tex <- c(tex, sprintf("        %s & %s & %s & %s & %s & %s\\\\",
        lab_beta(term), apa(co[i, "Estimate"]), apa(co[i, "Std. Error"]),
        apa(co[i, "z value"]), apa_p(co[i, "Pr(>|z|)"]),
        stars(co[i, "Pr(>|z|)"])))
}

# Random effects block: variance and SD, in the manuscript's layout.
small <- function(x, d = 3) ifelse(x < 10^(-d), sprintf("$< .%s1$", strrep("0", d - 1)), apa(x, d))
tex <- c(tex, "        \\midrule",
         "        \\multicolumn{2}{l}{Random effect variance}\\\\",
         "        Groups & Name & Variance & Std.Dev.& &\\\\")
pretty_grp <- c(dataset = "Dataset", seed = "Seed",
                "dataset:seed" = "Dataset:Seed")
for (nm in names(vcl)) {
    v <- as.numeric(vcl[[nm]])[1]
    gname <- if (!is.na(pretty_grp[nm])) pretty_grp[nm] else nm
    tex <- c(tex, sprintf("        %s & (Intercept) & %s & %s & &\\\\",
                          gname, small(v), small(sqrt(v))))
}
tex <- c(tex, "", "        \\bottomrule",
         "    %\\caption*{Significance levels: * $p<.05$, ** $p<.01$, *** $p<.001$.}",
         "\\end{longtable}", "}")
writeLines(tex, file.path(OUT, "sweep_table.tex"))
cat("wrote sweep_table.csv, sweep_table_random.csv, sweep_table.tex\n")

# --- per-dataset panels ------------------------------------------------------
# The submitted manuscript's main figure is a four-panel small multiple, one
# panel per dataset. The main-effect model above cannot fill it: its dataset
# random intercept is estimated at SD 0.015, so it predicts the SAME ratio
# profile in all four panels, while the observed profiles differ sharply
# (MNIST peaks at 0.2 and recovers at 1.0; notMNIST peaks at 0.3 and floors
# from 0.8). Those are dataset-specific shapes, not level shifts, so the panel
# figure needs ratio x dataset as a FIXED interaction. 11 x 4 cells at 5 seeds
# each supports it.
#
# Both models are kept: the main-effect fit is the pooled summary reported in
# the table, this one backs the figure and is tested against it below.
fit_sw_int <- glmmTMB(
    acc ~ ratio_f * dataset + (1 | seed) + (1 | dataset:seed),
    family = beta_family(link = "logit"), data = sw
)
cat("\n--- sweep, ratio x dataset interaction (backs the panel figure) ---\n")
cat("LRT against the main-effect model:\n")
print(anova(fit_sw, fit_sw_int))
write.csv(tidy_fit(fit_sw_int, "sweep_interaction"),
          file.path(OUT, "sweep_coefs_interaction.csv"), row.names = FALSE)

ndd <- expand.grid(ratio_f = levels(sw$ratio_f), dataset = levels(sw$dataset),
                   stringsAsFactors = FALSE)
ndd$seed <- NA
prd <- predict(fit_sw_int, newdata = ndd, type = "link", se.fit = TRUE,
               allow.new.levels = TRUE)
sweep_pred_ds <- data.frame(
    dataset = ndd$dataset,
    sleep_rate = as.numeric(as.character(ndd$ratio_f)),
    fit = inv(prd$fit),
    lo = inv(prd$fit - 1.96 * prd$se.fit),
    hi = inv(prd$fit + 1.96 * prd$se.fit)
)
sweep_pred_ds <- sweep_pred_ds[order(sweep_pred_ds$dataset, sweep_pred_ds$sleep_rate), ]
write.csv(sweep_pred_ds, file.path(OUT, "sweep_predictions_by_dataset.csv"),
          row.names = FALSE)


# =============================================================================
# 2. BASELINES — sleep against conventional stabilization (R3.1)
# =============================================================================
bl <- read.csv(file.path(REPO, "results", "baselines", "baselines_summary.csv"))
bl <- bl[!is.na(bl$test_accuracy), ]
bl$acc <- squeeze(bl$test_accuracy, 700)
# `none` as reference so each coefficient reads as that method's gain over
# unregularized STDP — the contrast the reviewer asked about.
bl$method <- relevel(factor(bl$method), ref = "none")
bl$dataset <- factor(bl$dataset); bl$seed <- factor(bl$seed)

fit_bl <- glmmTMB(
    acc ~ method + (1 | dataset) + (1 | seed) + (1 | dataset:seed),
    family = beta_family(link = "logit"), data = bl
)
report(fit_bl, "2. BASELINES  acc ~ method + (1|dataset) + (1|seed) + (1|dataset:seed)")

nd <- data.frame(method = levels(bl$method), dataset = NA, seed = NA)
pr <- predict(fit_bl, newdata = nd, type = "link", se.fit = TRUE,
              allow.new.levels = TRUE, re.form = NA)
bl_pred <- data.frame(method = nd$method, fit = inv(pr$fit),
                      lo = inv(pr$fit - 1.96 * pr$se.fit),
                      hi = inv(pr$fit + 1.96 * pr$se.fit))
write.csv(bl_pred, file.path(OUT, "baselines_predictions.csv"), row.names = FALSE)
write.csv(tidy_fit(fit_bl, "baselines"), file.path(OUT, "baselines_coefs.csv"), row.names = FALSE)

# The contrast the paper actually has to report: sleep vs the best conventional
# method. Refit with sleep as reference so it appears directly with a p-value.
bl$method_s <- relevel(bl$method, ref = "sleep")
fit_bl_s <- glmmTMB(
    acc ~ method_s + (1 | dataset) + (1 | seed) + (1 | dataset:seed),
    family = beta_family(link = "logit"), data = bl
)
cat("\n--- same fit, sleep as reference (sleep-vs-each contrasts) ---\n")
print(summary(fit_bl_s)$coefficients$cond)
write.csv(tidy_fit(fit_bl_s, "baselines_ref_sleep"),
          file.path(OUT, "baselines_coefs_ref_sleep.csv"), row.names = FALSE)

# =============================================================================
# 3. ABLATION — 2^4 component factorial (R3.2), MNIST only
# =============================================================================
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
report(fit_ab, "3. ABLATION  acc ~ downscale * noise * stdp * suppress + (1|seed)")
write.csv(tidy_fit(fit_ab, "ablation_full"),
          file.path(OUT, "ablation_coefs.csv"), row.names = FALSE)

# Conditions WITHOUT downscaling collapse to a degenerate state: three distinct
# configurations returned an identical multiset of accuracies across seeds,
# which is the signature of weights saturating against the clip bounds rather
# than of a graded effect. Main effects estimated over the full factorial are
# therefore dominated by that floor. Refit on the downscaling-present half,
# where weights stay in range and the remaining components are interpretable.
ab_on <- ab[ab$downscale == "on", ]
fit_ab_on <- glmmTMB(
    acc ~ noise * stdp * suppress + (1 | seed),
    family = beta_family(link = "logit"), data = ab_on
)
report(fit_ab_on,
       "3b. ABLATION, downscaling present  acc ~ noise * stdp * suppress + (1|seed)")
write.csv(tidy_fit(fit_ab_on, "ablation_downscale_on"),
          file.path(OUT, "ablation_coefs_downscale_on.csv"), row.names = FALSE)

# Per-condition predictions for the figure, over all 16 cells
nd <- unique(ab[, c("downscale", "noise", "stdp", "suppress")])
nd$seed <- NA
pr <- predict(fit_ab, newdata = nd, type = "link", se.fit = TRUE,
              allow.new.levels = TRUE, re.form = NA)
ab_pred <- cbind(nd[, 1:4],
                 fit = inv(pr$fit),
                 lo = inv(pr$fit - 1.96 * pr$se.fit),
                 hi = inv(pr$fit + 1.96 * pr$se.fit))
ab_pred$code <- apply(ab_pred[, c("downscale", "noise", "stdp", "suppress")], 1,
                      function(r) paste0(ifelse(r == "on", "1", "0"), collapse = ""))
write.csv(ab_pred, file.path(OUT, "ablation_predictions.csv"), row.names = FALSE)

cat("\nno-sleep reference (not in the factorial): mean acc",
    mean(ab_ref$test_accuracy), "over", nrow(ab_ref), "seeds\n")
write.csv(data.frame(mean_acc = mean(ab_ref$test_accuracy), n = nrow(ab_ref)),
          file.path(OUT, "ablation_nosleep_reference.csv"), row.names = FALSE)

cat("\nwrote coefficient and prediction tables to", OUT, "\n")
