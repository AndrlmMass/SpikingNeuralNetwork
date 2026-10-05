# =============================================================================
# ABLATION ACROSS DATASETS -- 2^4 component factorial x 4 datasets
# =============================================================================
#   accuracy ~ downscale * noise * stdp * suppress * dataset + (1 | seed)
#
# The component factorial is crossed with dataset, so every component effect
# and every component interaction is allowed to differ by dataset. With five
# seeds per cell this is the saturated cell-means model, equivalent to a
# 16-level condition factor crossed with dataset; the factorial form is kept so
# per-component contrasts can be read off directly.
#
# Inputs:  results/ablation/ablation_summary.csv                (MNIST, 85 cells)
#          results/ablation_datasets/ablation_summary.csv       (FMNIST, KMNIST,
#                                                                notMNIST, 255 cells)
#          Each includes 5 no-sleep reference runs per dataset (code "none").
# Outputs: results/glmm/ablation_ds_coefs.csv
#          results/glmm/ablation_ds_predictions.csv   fitted accuracy, all 64 cells
#          results/glmm/ablation_ds_marginal.csv      each component off vs on, by dataset
#          results/glmm/ablation_logor_table.csv      main effects + pairwise interactions (logit)
#          results/glmm/ablation_table.tex            the same, as the LaTeX table
#          results/glmm/ablation_ds_nosleep_reference.csv
#
# Usage (from the repo root):
#   Rscript src/glmm/fit_ablation_datasets.R
# =============================================================================

source("src/glmm/common.R")
suppressPackageStartupMessages(library(emmeans))

comps <- c("downscale", "noise", "stdp", "suppress")
read_ab <- function(p) {
    d <- read.csv(p, colClasses = c(code = "character"))
    d[, c("code", "dataset", "seed", comps, "test_accuracy", "sleep_rate")]
}
ab <- rbind(
    read_ab(file.path(REPO, "results", "ablation", "ablation_summary.csv")),
    read_ab(file.path(REPO, "results", "ablation_datasets", "ablation_summary.csv"))
)

# Guard the assumptions the model rests on rather than trusting them.
sleep_rates <- unique(ab$sleep_rate[ab$code != "none"])
stopifnot(length(sleep_rates) == 1)          # one shared ratio across all cells
cat("sleep ratio used by every factorial cell:", sleep_rates, "\n")
n_missing <- sum(is.na(ab$test_accuracy))
if (n_missing) cat("WARNING:", n_missing, "cells without an accuracy are dropped\n")
ab <- ab[!is.na(ab$test_accuracy), ]
tab <- table(ab$code, ab$dataset)
print(tab)
if (any(tab != 5)) cat("WARNING: not every condition x dataset cell has 5 seeds\n")

ref <- ab[ab$code == "none", ]
ab <- ab[ab$code != "none", ]
ab$acc <- squeeze(ab$test_accuracy, 700)
for (v in comps) {
    ab[[v]] <- factor(ifelse(tolower(as.character(ab[[v]])) %in% c("true", "1"),
                             "on", "off"), levels = c("off", "on"))
}
ab$dataset <- factor(ab$dataset, levels = c("mnist", "fmnist", "kmnist", "notmnist"))
ab$seed <- factor(ab$seed)

# Beta precision (phi) per component-combination x dataset cell, as in
# fit_baselines.R (phi ~ method * dataset). Seed-to-seed spread differs between
# collapsed (no downscaling, at chance) and working networks, and that differs
# by dataset, so a single phi pools unlike cells. Chosen by AIC on 2026-10-05
# (per cell -1207.4, phi ~ downscale * dataset -1194.3, shared phi -1036.1);
# BIC prefers phi ~ downscale * dataset. Estimates and conclusions were the same
# under every structure compared.
ab$cell <- interaction(ab$code, ab$dataset, drop = TRUE)

fit <- glmmTMB(
    acc ~ downscale * noise * stdp * suppress * dataset + (1 | seed),
    dispformula = ~ cell, family = beta_family(link = "logit"), data = ab
)
report(fit, "ABLATION x DATASET  acc ~ downscale*noise*stdp*suppress*dataset + (1|seed), phi ~ cell")
write.csv(tidy_fit(fit, "ablation_x_dataset"),
          file.path(OUT, "ablation_ds_coefs.csv"), row.names = FALSE)

# Fitted accuracy for every condition x dataset, on the accuracy scale.
reg <- regrid(emmeans(fit, ~ downscale * noise * stdp * suppress | dataset),
              transform = "response")
pred <- as.data.frame(confint(reg))
names(pred)[names(pred) == "response"] <- "fit"
names(pred)[names(pred) == "asymp.LCL"] <- "lo"
names(pred)[names(pred) == "asymp.UCL"] <- "hi"
pred$code <- apply(pred[, comps], 1,
                   function(r) paste0(ifelse(r == "on", "1", "0"), collapse = ""))
write.csv(pred[, c("dataset", "code", comps, "fit", "lo", "hi")],
          file.path(OUT, "ablation_ds_predictions.csv"), row.names = FALSE)

# Each component off vs on per dataset, averaged over the eight combinations of
# the other three components. Averaged on the accuracy scale (reg is already
# regridded), so "on" is the mean predicted accuracy of the eight cells with the
# component on. Feeds the slope figure (plot_glmms_bw.py, fig_ablation).
marg <- do.call(rbind, lapply(comps, function(v) {
    m <- as.data.frame(confint(emmeans(reg, as.formula(paste("~", v, "| dataset")))))
    data.frame(dataset = m$dataset, component = v, state = m[[v]],
               fit = m$response, lo = m$asymp.LCL, hi = m$asymp.UCL)
}))
write.csv(marg, file.path(OUT, "ablation_ds_marginal.csv"), row.names = FALSE)

# --- Log-odds-ratio table: main effects and pairwise interactions ------------
# Logit scale, averaged with equal weight over the four datasets and over the
# components not involved in the effect, in the layout of the baselines and
# sweep tables. One block per component, in the order of the slope figure: the
# first row is the component's main effect (on vs off), the indented rows its
# two-way interactions with the components listed after it, so each pair
# appears once. An interaction is the change in the component's log odds ratio
# when the other component is switched on:
#   (A on - A off | B on) - (A on - A off | B off).
# Unadjusted Wald tests, as in the baselines and sweep tables.
emm_options(msg.interaction = FALSE)
comp_order <- c("downscale", "stdp", "noise", "suppress")
comp_label <- c(downscale = "Downscaling", stdp = "STDP", noise = "Membrane noise",
                suppress = "Input suppression")
tidy_ctr <- function(ctr, block, term) {
    s <- as.data.frame(summary(ctr, infer = c(FALSE, TRUE)))
    stopifnot(nrow(s) == 1)
    data.frame(block = block, term = term, estimate = s$estimate, std_error = s$SE,
               z = s$z.ratio, p = s$p.value)
}
lor <- NULL
for (i in seq_along(comp_order)) {
    a <- comp_order[i]
    lor <- rbind(lor, tidy_ctr(
        contrast(emmeans(fit, as.formula(paste("~", a))), "revpairwise"), a, "main"))
    for (b in comp_order[-seq_len(i)]) {
        lor <- rbind(lor, tidy_ctr(
            contrast(emmeans(fit, as.formula(paste("~", a, "*", b))),
                     interaction = c("revpairwise", "revpairwise")), a, b))
    }
}

stars <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "")))
lor$sig <- stars(lor$p)
print(lor, digits = 4)
write.csv(lor, file.path(OUT, "ablation_logor_table.csv"), row.names = FALSE)

vcl <- glmmTMB::VarCorr(fit)$cond
re <- data.frame(group = names(vcl),
                 variance = sapply(vcl, function(v) as.numeric(v)[1]), row.names = NULL)
re$sd <- sqrt(re$variance)

apa <- function(x, d = 3) {
    x <- round(x, d) + 0                     # + 0 turns -0 into 0 (no "-.000")
    out <- sprintf(paste0("%.", d, "f"), x)
    out <- sub("^0\\.", ".", out)
    sub("^-0\\.", "-.", out)
}
apa_p <- function(p) ifelse(p < 0.001, "$<.001$", apa(p, 3))
small <- function(x, d = 3) ifelse(x < 10^(-d), sprintf("$< .%s1$", strrep("0", d - 1)), apa(x, d))
tex_row <- function(lab, r) sprintf("        %s & %s & %s & %s & %s & %s\\\\",
    lab, apa(r$estimate), apa(r$std_error), apa(r$z), apa_p(r$p), r$sig)
term_label <- c(stdp = "STDP", noise = "membrane noise", suppress = "input suppression")
header <- "    Effect  & Estimate & Std. Error & \\textit{z} value & \\textit{p}-value & \\\\"
tex <- c("{", "\\setlength{\\tabcolsep}{10pt} % Only affects this table",
         "\\begin{longtable}{lrrrrl}",
         paste0("    \\caption{Mixed effect model with beta regression: main effects and ",
                "pairwise interactions of the sleep components \\label{tab:betareg_ablation}}\\\\"),
         "    \\toprule", header, "    \\midrule", "    \\endfirsthead", "",
         "    \\toprule", header, "    \\midrule", "    \\endhead")
for (i in seq_along(comp_order)) {
    a <- comp_order[i]
    b <- lor[lor$block == a, ]
    if (i > 1) tex <- c(tex, "        \\midrule")
    tex <- c(tex, sprintf("        \\multicolumn{6}{l}{\\textit{%s}}\\\\", comp_label[[a]]))
    for (k in seq_len(nrow(b))) {
        lab <- if (b$term[k] == "main") "Main effect (on vs off)" else
            sprintf("\\quad $\\times$ %s", term_label[[b$term[k]]])
        tex <- c(tex, tex_row(lab, b[k, ]))
    }
}
pretty_grp <- c(seed = "Seed")
tex <- c(tex, "        \\midrule",
         "        \\multicolumn{2}{l}{Random effect variance}\\\\",
         "        Groups & Name & Variance & Std.Dev.& &\\\\")
for (k in seq_len(nrow(re))) {
    tex <- c(tex, sprintf("        %s & (Intercept) & %s & %s & &\\\\",
                          pretty_grp[[re$group[k]]], small(re$variance[k]), small(re$sd[k])))
}
tex <- c(tex, "", "        \\bottomrule",
         paste0("    \\multicolumn{6}{p{0.9\\linewidth}}{\\footnotesize Log odds ratios on the logit ",
                "scale. Main effects compare each component on versus off, averaged over the four ",
                "datasets and over all combinations of the other components; interactions give the ",
                "change in the first component's log odds ratio when the second component is also ",
                "switched on, and each pair is reported once. ",
                "Model: accuracy $\\sim$ downscaling $\\times$ STDP $\\times$ membrane noise ",
                "$\\times$ input suppression $\\times$ dataset + (1$|$seed), with the Beta precision ",
                "$\\phi$ estimated separately for each combination of components and dataset. ",
                "Significance levels: * $p<.05$, ** $p<.01$, *** $p<.001$.}\\\\"),
         "\\end{longtable}", "}")
writeLines(tex, file.path(OUT, "ablation_table.tex"))
cat("wrote ablation_table.tex, ablation_logor_table.csv\n")

refs <- aggregate(test_accuracy ~ dataset, data = ref, FUN = mean)
names(refs)[2] <- "mean_acc"
refs$n <- aggregate(test_accuracy ~ dataset, data = ref, FUN = length)$test_accuracy
print(refs)
write.csv(refs, file.path(OUT, "ablation_ds_nosleep_reference.csv"), row.names = FALSE)

cat("\nwrote ablation_ds_{coefs,predictions,marginal,nosleep_reference}.csv\n")
