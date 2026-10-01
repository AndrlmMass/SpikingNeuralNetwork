# =============================================================================
# ABLATION ACROSS DATASETS -- 2^4 component factorial x 4 datasets
# =============================================================================
#   accuracy ~ downscale * noise * stdp * suppress * dataset + (1 | seed)
#
# The component factorial is crossed with dataset, so every component effect
# and every component interaction is allowed to differ by dataset. With five
# seeds per cell this is the saturated cell-means model, equivalent to a
# 16-level condition factor crossed with dataset; the factorial form is kept so
# per-component contrasts can be read off directly. Same family and link as
# fit_ablation.R (Beta, logit), so the MNIST numbers stay comparable.
#
# Inputs:  results/ablation/ablation_summary_mnist_backup.csv   (MNIST, 85 cells)
#          results/ablation_datasets/ablation_summary.csv       (FMNIST, KMNIST,
#                                                                notMNIST, 255 cells)
# Outputs: results/glmm/ablation_ds_coefs.csv
#          results/glmm/ablation_ds_predictions.csv   fitted accuracy, all 64 cells
#          results/glmm/ablation_ds_paths.csv         add / remove effects by dataset
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
    read_ab(file.path(REPO, "results", "ablation", "ablation_summary_mnist_backup.csv")),
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

fit <- glmmTMB(
    acc ~ downscale * noise * stdp * suppress * dataset + (1 | seed),
    family = beta_family(link = "logit"), data = ab
)
report(fit, "ABLATION x DATASET  acc ~ downscale*noise*stdp*suppress*dataset + (1|seed)")
write.csv(tidy_fit(fit, "ablation_x_dataset"),
          file.path(OUT, "ablation_ds_coefs.csv"), row.names = FALSE)

# Omnibus: does the component structure differ between datasets at all?
fit0 <- glmmTMB(
    acc ~ downscale * noise * stdp * suppress + dataset + (1 | seed),
    family = beta_family(link = "logit"), data = ab
)
cat("\nLikelihood-ratio test, components x dataset interaction:\n")
lrt <- anova(fit0, fit)
print(lrt)
lrt_row <- data.frame(chisq = lrt$Chisq[2], df = lrt[["Chi Df"]][2], p = lrt[["Pr(>Chisq)"]][2])
write.csv(lrt_row, file.path(OUT, "ablation_ds_lrt.csv"), row.names = FALSE)

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

# Single-path effects per dataset, as in fit_ablation_marginal.R:
#   add    = cell(only v on)          - cell(all off)
#   remove = cell(all on except v)    - cell(all on)
# Coefficients index the 16 conditions within one dataset, in grid order; take
# that order from the grid itself rather than assuming it.
g <- as.data.frame(reg)
g <- g[g$dataset == levels(ab$dataset)[1], ]
stopifnot(nrow(g) == 16)
idx <- function(on) {
    which(Reduce(`&`, lapply(comps, function(v) (g[[v]] == "on") == (v %in% on))))
}
coef <- function(plus, minus) {
    k <- numeric(nrow(g)); k[idx(plus)] <- 1; k[idx(minus)] <- -1; k
}
ctr <- list()
for (v in comps) {
    ctr[[paste0("add_", v)]] <- coef(v, character(0))
    ctr[[paste0("remove_", v)]] <- coef(setdiff(comps, v), comps)
}
paths <- as.data.frame(summary(contrast(
    emmeans(reg, ~ downscale * noise * stdp * suppress | dataset), ctr),
    infer = c(TRUE, TRUE)))
paths$kind <- sub("_.*", "", paths$contrast)
paths$component <- sub("^[a-z]+_", "", paths$contrast)
paths <- paths[, c("dataset", "kind", "component", "estimate", "asymp.LCL", "asymp.UCL", "p.value")]
names(paths) <- c("dataset", "kind", "component", "est", "lo", "hi", "p")
print(paths, digits = 3)
write.csv(paths, file.path(OUT, "ablation_ds_paths.csv"), row.names = FALSE)

# --- LaTeX table: add / remove effects per component and dataset -------------
# Accuracy-scale differences (regridded), 95% CI, unadjusted Wald p as in the
# baselines table. Same layout conventions as baselines_table.tex.
apa <- function(x, d = 3) {
    x <- round(x, d) + 0                     # + 0 turns -0 into 0 (no "-.000")
    out <- sprintf(paste0("%.", d, "f"), x)
    out <- sub("^0\\.", ".", out)
    sub("^-0\\.", "-.", out)
}
stars <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "")))
cell <- function(ds, kind, comp) {
    r <- paths[paths$dataset == ds & paths$kind == kind & paths$component == comp, ]
    s <- stars(r$p)
    sprintf("%s [%s, %s]%s", apa(r$est), apa(r$lo), apa(r$hi),
            if (nchar(s)) paste0("$^{", s, "}$") else "")
}
ds_order <- c("mnist", "kmnist", "fmnist", "notmnist")
comp_label <- c(downscale = "Downscaling", stdp = "STDP", noise = "Membrane noise",
                suppress = "Input suppression")
kind_label <- c(add = "add (all others off)", remove = "remove (all others on)")
header <- "    Component & MNIST & KMNIST & Fashion-MNIST & NotMNIST \\\\"
tex <- c("{", "\\setlength{\\tabcolsep}{6pt} % Only affects this table",
         "\\begin{longtable}{lllll}",
         paste0("    \\caption{Mixed effect model with beta regression: change in test accuracy ",
                "from adding or removing each sleep component, per dataset ",
                "\\label{tab:betareg_ablation}}\\\\"),
         "    \\toprule", header, "    \\midrule", "    \\endfirsthead", "",
         "    \\toprule", header, "    \\midrule", "    \\endhead")
for (comp in names(comp_label)) {
    tex <- c(tex, sprintf("        \\multicolumn{5}{l}{\\textit{%s}}\\\\", comp_label[[comp]]))
    for (kind in names(kind_label)) {
        cells <- sapply(ds_order, function(d) cell(d, kind, comp))
        tex <- c(tex, sprintf("        \\quad %s & %s\\\\", kind_label[[kind]],
                              paste(cells, collapse = " & ")))
    }
    if (comp != tail(names(comp_label), 1)) tex <- c(tex, "        \\midrule")
}
p_lrt <- if (lrt_row$p < 0.001) "$p<.001$" else sprintf("$p=%s$", apa(lrt_row$p))
tex <- c(tex, "", "        \\bottomrule",
         paste0("    \\multicolumn{5}{p{0.95\\linewidth}}{\\footnotesize Differences in predicted ",
                "test accuracy with 95\\% confidence intervals. \\textit{add}: the component alone ",
                "minus all components off; \\textit{remove}: all components minus all but this ",
                "component. Model: accuracy $\\sim$ downscale $\\times$ noise $\\times$ STDP ",
                "$\\times$ suppress $\\times$ dataset + (1$|$seed). The components $\\times$ dataset ",
                sprintf("interaction is significant ($\\chi^2(%d) = %.1f$, %s). ",
                        as.integer(lrt_row$df), lrt_row$chisq, p_lrt),
                "Without downscaling every condition is at chance, so \\textit{add} effects of the ",
                "other components cannot fall below zero. ",
                "Significance levels: * $p<.05$, ** $p<.01$, *** $p<.001$.}\\\\"),
         "\\end{longtable}", "}")
writeLines(tex, file.path(OUT, "ablation_table.tex"))
cat("wrote ablation_table.tex\n")

refs <- aggregate(test_accuracy ~ dataset, data = ref, FUN = mean)
names(refs)[2] <- "mean_acc"
refs$n <- aggregate(test_accuracy ~ dataset, data = ref, FUN = length)$test_accuracy
print(refs)
write.csv(refs, file.path(OUT, "ablation_ds_nosleep_reference.csv"), row.names = FALSE)

cat("\nwrote ablation_ds_{coefs,predictions,paths,nosleep_reference}.csv\n")
