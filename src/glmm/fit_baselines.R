# =============================================================================
# BASELINES — sleep against conventional stabilization (R3.1)
# =============================================================================
#   accuracy ~ method + (1 | dataset) + (1 | seed) + (1 | dataset:seed)
#   5 methods, 4 datasets, 5 seeds; Beta family, logit link, one shared
#   precision parameter.
#
# Method is a fixed factor with sleep as the reference level, so every reported
# effect is a method minus sleep. Dataset, seed and their combination are random
# intercepts (each dataset x seed pair is one training setup shared by all five
# methods), matching the structure of the submitted article. Reported effects
# are population-level: a typical dataset and seed (random effects at zero).
#
# Usage (from the repo root):
#   Rscript src/glmm/fit_baselines.R
#
# Inputs:  results/baselines/baselines_summary.csv
# Outputs: results/glmm/baselines_coefs.csv
#          results/glmm/baselines_predictions.csv             (population level)
#          results/glmm/baselines_predictions_by_dataset.csv   (figure bars)
#          results/glmm/baselines_table.{csv,tex}, baselines_table_random.csv
# =============================================================================

source("src/glmm/common.R")

bl <- read.csv(file.path(REPO, "results", "baselines", "baselines_summary.csv"))
bl <- bl[!is.na(bl$test_accuracy), ]
bl$acc <- squeeze(bl$test_accuracy, N_TEST)
bl$method <- relevel(factor(bl$method), ref = "sleep")
bl$dataset <- factor(bl$dataset)
bl$seed <- factor(bl$seed)

fit_bl <- glmmTMB(acc ~ method + (1 | dataset) + (1 | seed) + (1 | dataset:seed),
                  family = beta_family(link = "logit"), data = bl)
report(fit_bl, "BASELINES  acc ~ method + (1|dataset) + (1|seed) + (1|dataset:seed)")
write.csv(tidy_fit(fit_bl, "baselines"), file.path(OUT, "baselines_coefs.csv"),
          row.names = FALSE)

# =============================================================================
# Population-level estimates (predict(); random effects at zero)
# =============================================================================
pred <- pop_pred(fit_bl, data.frame(method = factor(levels(bl$method),
                                                    levels = levels(bl$method))))
write.csv(data.frame(method = as.character(pred$method), fit = pred$fit,
                     lo = pred$lo, hi = pred$hi),
          file.path(OUT, "baselines_predictions.csv"), row.names = FALSE)

# Sleep (intercept) and each method vs sleep (coefficients), Wald z tests.
cf <- coef_rows(fit_bl, "method")

# Dataset-specific predictions for the figure: fixed effects plus each dataset's
# random intercept, seed at zero. No per-dataset tests are made.
pred_ds <- cond_pred(fit_bl, expand.grid(method = levels(bl$method),
                                         dataset = levels(bl$dataset)), bl$seed)
pred_ds <- data.frame(dataset = as.character(pred_ds$dataset),
                      method = as.character(pred_ds$method),
                      fit = pred_ds$fit, lo = pred_ds$lo, hi = pred_ds$hi)
write.csv(pred_ds[order(pred_ds$dataset, pred_ds$method), ],
          file.path(OUT, "baselines_predictions_by_dataset.csv"), row.names = FALSE)

# =============================================================================
# Regression table
# =============================================================================
# First row: sleep's accuracy (logit scale); then every other method minus sleep.
# The row label carries the ratio the sleep arm actually ran at (from the sweep).
sleep_pct <- round(100 * max(bl$sleep_ratio[as.character(bl$method) == "sleep"], na.rm = TRUE))
label <- c(sleep = sprintf("Sleep %d\\%%", sleep_pct), none = "No regularization",
           decay = "Continuous decay", norm_layer = "Layer normalization",
           norm_neuron = "Synaptic scaling")
order_terms <- c("none", "decay", "norm_layer", "norm_neuron")
stars <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**",
                     ifelse(p < 0.05, "*", "")))

cf$level[cf$level == "(Intercept)"] <- "sleep"
cf <- cf[match(c("sleep", order_terms), cf$level), ]
stopifnot(!anyNA(cf$level))
tab <- data.frame(term = cf$level, estimate = cf$estimate, std_error = cf$SE,
                  z = cf$z.ratio, p = cf$p.value)
rownames(tab) <- NULL
tab$sig <- stars(tab$p)

re <- random_effects(fit_bl)
write.csv(tab, file.path(OUT, "baselines_table.csv"), row.names = FALSE)
write.csv(re, file.path(OUT, "baselines_table_random.csv"), row.names = FALSE)

# --- LaTeX, same layout as sweep_table.tex -----------------------------------
tex_row <- function(lab, r) sprintf("        %s & %s & %s & %s & %s & %s\\\\",
    lab, apa(r$estimate), apa(r$std_error), apa(r$z), apa_p(r$p), r$sig)
header <- "    Effect  & Estimate & Std. Error & \\textit{z} value & \\textit{p}-value & \\\\"
tex <- c(
    "{",
    "\\setlength{\\tabcolsep}{10pt} % Only affects this table",
    "\\begin{longtable}{lrrrrl}",
    "    \\caption{Mixed effect model with beta regression: sleep against conventional stabilization \\label{tab:betareg_baselines}}\\\\",
    "    \\toprule", header, "    \\midrule", "    \\endfirsthead", "",
    "    \\toprule", header, "    \\midrule", "    \\endhead"
)
for (i in seq_len(nrow(tab))) {
    lab <- if (tab$term[i] == "sleep") label[["sleep"]] else
        paste0("\\quad ", label[[tab$term[i]]])
    tex <- c(tex, tex_row(lab, tab[i, ]))
}
tex <- c(tex, re_tex(re), "", "        \\bottomrule",
         paste0("    \\multicolumn{6}{p{0.9\\linewidth}}{\\footnotesize Estimates on the ",
                "logit scale. The first row is sleep's accuracy and the indented rows ",
                "are each regularizer minus sleep. Model: accuracy $\\sim$ method + ",
                "(1$|$dataset) + (1$|$seed) + (1$|$dataset:seed), with a single ",
                "precision parameter; estimates are for a typical dataset and seed. ",
                "Significance levels: * $p<.05$, ** $p<.01$, *** $p<.001$.}\\\\"),
         "\\end{longtable}", "}")
writeLines(tex, file.path(OUT, "baselines_table.tex"))

cat("\n--- table (logit scale, population level) ---\n")
print(tab, digits = 4)
print(re, digits = 4)
cat("\nwrote baselines_coefs.csv, baselines_predictions{,_by_dataset}.csv,",
    "baselines_table.{csv,tex}, baselines_table_random.csv\n")
