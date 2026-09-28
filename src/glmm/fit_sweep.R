# =============================================================================
# SWEEP — accuracy as a function of sleep duration, per dataset
# =============================================================================
#   accuracy ~ sleep duration x dataset (11 levels, 4 datasets, 5 seeds)
#
# Usage (from the repo root):
#   Rscript src/glmm/fit_sweep.R
#
# Inputs:  results/sweep/sweep_summary.csv
# Outputs: results/glmm/sweep_coefs.csv
#          results/glmm/sweep_predictions.csv             (dataset-averaged)
#          results/glmm/sweep_predictions_by_dataset.csv   (figure panels)
#          results/glmm/sweep_contrasts_by_dataset.csv     (simple effects)
#          results/glmm/sweep_table.{csv,tex}, sweep_table_random.csv
#          plus a console summary of the fit
# =============================================================================

source("src/glmm/common.R")
suppressPackageStartupMessages(library(emmeans))

# emmeans applies a multiplicity adjustment to trt.vs.ctrl1 by default (Dunnett).
# The submitted table reports unadjusted Wald tests, so every contrast below
# passes adjust = "none" to stay comparable. Set once, explicitly, rather than
# relying on the argument being remembered at each call site.
emm_options(msg.interaction = FALSE, msg.nesting = FALSE)

sw <- read.csv(file.path(REPO, "results", "sweep", "sweep_summary.csv"))
sw <- sw[!is.na(sw$test_accuracy), ]
sw$n_test <- 700   # fixed within dataset; used only for the (0,1) squeeze
sw$acc <- squeeze(sw$test_accuracy, sw$n_test)

# Sleep duration as a FACTOR, not continuous: the manuscript reports one
# coefficient per level against 0%, and the response is non-monotonic, so a
# linear term would misrepresent it. Treatment-coded, 0% as reference.
sw$ratio_f <- relevel(factor(sw$sleep_rate), ref = "0")
sw$dataset <- factor(sw$dataset)
sw$seed <- factor(sw$seed)

# Sum-to-zero (deviation) coding on dataset, so that the raw ratio_f
# coefficients read as the effect of sleep for an AVERAGE dataset rather than
# the effect at whichever dataset sorts first, which is what R's default
# treatment coding gives. This affects only the coefficient dump in
# sweep_coefs.csv: everything else below comes from emmeans, which works from
# the fitted cell means and gives identical results under either coding.
contrasts(sw$dataset) <- contr.sum(nlevels(sw$dataset))

# One model backs both the table and the figure. Dataset is a FIXED effect
# crossed with ratio, not a random intercept: there are only four datasets, too
# few to estimate a variance, they were chosen rather than sampled, and their
# ratio profiles differ in shape (MNIST peaks at 0.2 and recovers at 1.0;
# notMNIST peaks at 0.3 and floors from 0.8), which a random intercept -- a
# level shift -- cannot represent. The submitted manuscript used
# ratio + (1|dataset).
#
# (1|dataset:seed) is not used: it was estimated at SD ~0 (singular) and
# left the fixed effects unchanged to 1e-5.
fit_sw <- glmmTMB(
    acc ~ ratio_f * dataset + (1 | seed),
    family = beta_family(link = "logit"), data = sw
)
report(fit_sw, "SWEEP  acc ~ factor(sleep duration) * dataset + (1|seed)")
write.csv(tidy_fit(fit_sw, "sweep"), file.path(OUT, "sweep_coefs.csv"), row.names = FALSE)

# =============================================================================
# Estimated marginal means and contrasts (emmeans)
# =============================================================================
# Everything reported comes from these grids. emmeans averages on the link
# scale, so a dataset-averaged value back-transforms to the accuracy of a
# typical dataset rather than to the mean of four accuracies. Random effects sit
# at zero throughout, i.e. a typical seed.
emm_rd <- emmeans(fit_sw, ~ ratio_f * dataset)   # all 44 cells
emm_r <- emmeans(fit_sw, ~ ratio_f)              # averaged over datasets

# Predictions for the figure panels and for the dataset-averaged curve.
pred_out <- function(emm, ...) {
    s <- summary(emm, type = "response", infer = c(TRUE, FALSE))
    data.frame(..., sleep_rate = as.numeric(as.character(s$ratio_f)),
               fit = s$response, lo = s$asymp.LCL, hi = s$asymp.UCL)
}
s_rd <- summary(emm_rd, type = "response", infer = c(TRUE, FALSE))
sweep_pred_ds <- data.frame(
    dataset = as.character(s_rd$dataset),
    sleep_rate = as.numeric(as.character(s_rd$ratio_f)),
    fit = s_rd$response, lo = s_rd$asymp.LCL, hi = s_rd$asymp.UCL
)
sweep_pred_ds <- sweep_pred_ds[order(sweep_pred_ds$dataset, sweep_pred_ds$sleep_rate), ]
write.csv(sweep_pred_ds, file.path(OUT, "sweep_predictions_by_dataset.csv"),
          row.names = FALSE)

sweep_pred <- pred_out(emm_r)
sweep_pred <- sweep_pred[order(sweep_pred$sleep_rate), ]
write.csv(sweep_pred, file.path(OUT, "sweep_predictions.csv"), row.names = FALSE)

# --- the three contrast families --------------------------------------------
# 1. Averaged effect of each sleep duration vs 0%, over datasets (main text).
ctr <- as.data.frame(contrast(emm_r, "trt.vs.ctrl1", adjust = "none"))

# 2. Simple effects: each sleep duration vs 0% WITHIN one dataset, with its own
#    SE (appendix). This is the table that actually shows the divergence. The
#    interaction coefficients cannot be read this way -- they are deviations
#    from the average, so recovering a dataset's effect from them needs a sum of
#    two terms and a covariance no table prints.
ctr_ds <- as.data.frame(contrast(emm_rd, "trt.vs.ctrl1", by = "dataset",
                                 adjust = "none"))

# No interaction-coefficient family is computed. The deviation of each dataset's
# sleep effect from the average is exactly (2) minus (1), so it carries nothing
# the two tables above do not already say, and it is the form a reader cannot
# interpret without undoing the subtraction.

# Recover the numeric sleep level from the emmeans contrast label, which reads
# "ratio_f0.1 - ratio_f0". Drop everything from the minus sign on, then the
# factor-name prefix.
rate_of <- function(x) {
    lab <- sub(" *-.*$", "", as.character(x))
    out <- as.numeric(sub("^[^0-9.]*", "", lab))
    stopifnot(!any(is.na(out)))
    out
}
ctr$sleep_rate <- rate_of(ctr$contrast)
ctr_ds$sleep_rate <- rate_of(ctr_ds$contrast)

ctr_ds <- ctr_ds[order(ctr_ds$dataset, ctr_ds$sleep_rate),
                 c("dataset", "sleep_rate", "estimate", "SE", "z.ratio", "p.value")]
names(ctr_ds) <- c("dataset", "sleep_rate", "estimate", "std_error", "z", "p")
write.csv(ctr_ds, file.path(OUT, "sweep_contrasts_by_dataset.csv"), row.names = FALSE)

# =============================================================================
# Regression table
# =============================================================================
# A conventional GLMM table on the link scale: the reference cell, the averaged
# sleep effects, then the random-effect SD. Dataset does not appear -- the
# per-dataset picture lives in sweep_contrasts_by_dataset.csv (simple effects)
# and sweep_predictions_by_dataset.csv (absolute accuracies), both of which a
# reader can use directly. No back-transformed accuracy column here; those
# belong in the prediction files.
nz <- sort(unique(ctr$sleep_rate))

# Significance markers and p-value formatting, used only by this script.
stars <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**",
                     ifelse(p < 0.05, "*", ifelse(p < 0.1, ".", ""))))
fmt_p <- function(p) ifelse(p < 1e-4, "< 0.0001", sprintf("%.4f", p))

grab <- function(df) df[, c("estimate", "SE", "z.ratio", "p.value")]
names_std <- c("estimate", "std_error", "z", "p")

ref0 <- as.data.frame(summary(emm_r, infer = c(FALSE, TRUE)))
ref0 <- data.frame(estimate = ref0$emmean[ref0$ratio_f == "0"],
                   SE = ref0$SE[ref0$ratio_f == "0"],
                   z.ratio = ref0$z.ratio[ref0$ratio_f == "0"],
                   p.value = ref0$p.value[ref0$ratio_f == "0"])

tab <- rbind(
    cbind(block = "reference", sleep_rate = 0,
          term = "0\\% sleep", setNames(grab(ref0), names_std)),
    cbind(block = "sleep", sleep_rate = ctr$sleep_rate,
          term = sprintf("Sleep duration %g\\%%", ctr$sleep_rate * 100),
          setNames(grab(ctr), names_std))
)
rownames(tab) <- NULL
tab$sig <- stars(tab$p)

vcl <- glmmTMB::VarCorr(fit_sw)$cond
re <- data.frame(
    term = sprintf("SD (%s)", names(vcl)),
    sd = sapply(names(vcl), function(nm) sqrt(as.numeric(vcl[[nm]])[1])),
    row.names = NULL
)

write.csv(tab[, c("block", "sleep_rate", "term",
                  "estimate", "std_error", "z", "p", "sig")],
          file.path(OUT, "sweep_table.csv"), row.names = FALSE)
write.csv(re, file.path(OUT, "sweep_table_random.csv"), row.names = FALSE)

# --- console ---
bar <- strrep("=", 72)
cat("\n", bar, "\n",
    "MAIN RUN -- Beta GLMM (logit link)\n",
    "acc ~ factor(sleep duration) * dataset + (1|seed)\n",
    "sleep rows averaged over datasets\n",
    bar, "\n", sep = "")
cat(sprintf("%-30s %9s %8s %7s %11s\n", "Effect", "Estimate", "SE", "z", "p"))
plain <- function(x) gsub("\\\\", "", x)
for (i in seq_len(nrow(tab))) {
    if (i == 1 || tab$block[i] != tab$block[i - 1]) {
        cat(strrep("-", 72), "\n", sep = "")
    }
    cat(sprintf("%-30s %9.4f %8.4f %7.2f %8s %-3s\n",
                plain(tab$term[i]), tab$estimate[i], tab$std_error[i],
                tab$z[i], fmt_p(tab$p[i]), tab$sig[i]))
}
cat(strrep("-", 72), "\n", sep = "")
cat(sprintf("%-30s %9s\n", "Random effects", "SD"))
for (i in seq_len(nrow(re))) {
    cat(sprintf("%-30s %9.4f\n", re$term[i], re$sd[i]))
}
cat(strrep("-", 72), "\n", sep = "")
cat(sprintf("%-30s %9.2f\n", "Dispersion (phi)", sigma(fit_sw)))
cat(sprintf("%-30s %9d\n", "Observations", nrow(sw)))
cat(sprintf("%-30s %9d\n", "Datasets", nlevels(sw$dataset)))
cat(sprintf("%-30s %9d\n", "Seeds", nlevels(sw$seed)))
cat(sprintf("%-30s %9.1f\n", "AIC", AIC(fit_sw)))
cat(bar, "\n", sep = "")
cat("Signif.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1\n")
cat("Log-odds, averaged over the four datasets. Per-dataset effects are in\n",
    "sweep_contrasts_by_dataset.csv.\n", sep = "")

# Which dataset-averaged sleep rows average across a SIGN REVERSAL. An averaged
# effect near zero can mean either "small everywhere" or "large in both
# directions, cancelling"; only the per-dataset simple effects distinguish the
# two, so flag the rows the main-text table must not be read as a null.
cat("\n", bar, "\n", "Sign agreement of the per-dataset simple effects\n",
    bar, "\n", sep = "")
cat(sprintf("%-8s %9s %s\n", "ratio", "averaged", "per-dataset signs"))
for (r in nz) {
    e <- ctr_ds$estimate[ctr_ds$sleep_rate == r]
    flag <- if (max(e) > 0 && min(e) < 0) "CROSSOVER" else "same sign"
    cat(sprintf("%-8g %9.3f %s\n", r, ctr$estimate[ctr$sleep_rate == r], flag))
}
cat(bar, "\n", sep = "")

# =============================================================================
# LaTeX, matching the manuscript's existing beta-regression table
# =============================================================================
# longtable with a repeated header, beta_{j=..} row labels, a separate
# significance column, and APA number style (no leading zero). The submitted
# version carried an SG-SNN main effect and a sleep x model interaction; that
# model is dropped from the revision, and the dataset main effect and the
# sleep x dataset interaction take the same places in the layout.
apa <- function(x, d = 3) {
    out <- sprintf(paste0("%.", d, "f"), x)
    out <- sub("^0\\.", ".", out)
    sub("^-0\\.", "-.", out)
}
apa_p <- function(p) ifelse(p < 0.001, "$<.001$", apa(p, 3))
tex_row <- function(label, i) sprintf("        %s & %s & %s & %s & %s & %s\\\\",
    label, apa(tab$estimate[i]), apa(tab$std_error[i]), apa(tab$z[i]),
    apa_p(tab$p[i]), tab$sig[i])
subhead <- function(txt) sprintf("        \\multicolumn{6}{l}{\\textit{%s}}\\\\", txt)

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

i0 <- which(tab$block == "reference")
tex <- c(tex, sub("\\\\\\\\$", "\\\\\\\\[6pt]", tex_row("0\\% sleep", i0)))

tex <- c(tex, subhead("Main effect of sleep duration (vs.\\ 0\\%, averaged over datasets)"))
for (i in which(tab$block == "sleep")) {
    tex <- c(tex, tex_row(sprintf("$\\beta_{j=%g}$", tab$sleep_rate[i] * 100), i))
}

# Random effects block: variance and SD, in the manuscript's layout.
small <- function(x, d = 3) ifelse(x < 10^(-d), sprintf("$< .%s1$", strrep("0", d - 1)), apa(x, d))
tex <- c(tex, "        \\midrule",
         "        \\multicolumn{2}{l}{Random effect variance}\\\\",
         "        Groups & Name & Variance & Std.Dev.& &\\\\")
pretty_grp <- c(dataset = "Dataset", seed = "Seed")
for (nm in names(vcl)) {
    v <- as.numeric(vcl[[nm]])[1]
    gname <- if (!is.na(pretty_grp[nm])) pretty_grp[nm] else nm
    tex <- c(tex, sprintf("        %s & (Intercept) & %s & %s & &\\\\",
                          gname, small(v), small(sqrt(v))))
}
tex <- c(tex, "", "        \\bottomrule",
         paste0("    \\multicolumn{6}{p{0.9\\linewidth}}{\\footnotesize Estimates on the ",
                "logit scale, averaged over the four datasets. Effects within each ",
                "dataset are reported separately in the appendix.}\\\\"),
         "    %\\caption*{Significance levels: * $p<.05$, ** $p<.01$, *** $p<.001$.}",
         "\\end{longtable}", "}")
writeLines(tex, file.path(OUT, "sweep_table.tex"))
cat("\nwrote sweep_coefs.csv, sweep_table.csv, sweep_table_random.csv,",
    "sweep_table.tex,\n  sweep_predictions.csv, sweep_predictions_by_dataset.csv,",
    "sweep_contrasts_by_dataset.csv\n")
