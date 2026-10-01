# =============================================================================
# BASELINES — sleep against conventional stabilization (R3.1)
# =============================================================================
#   accuracy ~ regularization method x dataset (5 methods, 4 datasets, 5 seeds)
#
# Same structure as fit_sweep.R. Dataset is a FIXED effect crossed with method,
# not a random intercept: there are only four datasets, and the methods do not
# shift by a constant across them -- continuous decay was calibrated on MNIST
# and works only there (0.74 on MNIST, 0.11-0.29 elsewhere), which a random
# intercept cannot represent. The additive model is kept only for the
# likelihood-ratio test that justifies the interaction.
#
# Sleep is the reference level: every reported effect is a method minus sleep.
#
# Usage (from the repo root):
#   Rscript src/glmm/fit_baselines.R
#
# Inputs:  results/baselines/baselines_summary.csv
# Outputs: results/glmm/baselines_coefs.csv
#          results/glmm/baselines_predictions.csv             (dataset-averaged)
#          results/glmm/baselines_predictions_by_dataset.csv   (figure bars)
#          results/glmm/baselines_contrasts_by_dataset.csv     (simple effects)
#          results/glmm/baselines_table.{csv,tex}, baselines_table_random.csv
# =============================================================================

source("src/glmm/common.R")
suppressPackageStartupMessages(library(emmeans))
emm_options(msg.interaction = FALSE, msg.nesting = FALSE)

bl <- read.csv(file.path(REPO, "results", "baselines", "baselines_summary.csv"))
bl <- bl[!is.na(bl$test_accuracy), ]
bl$acc <- squeeze(bl$test_accuracy, 700)
bl$method <- relevel(factor(bl$method), ref = "sleep")
bl$dataset <- factor(bl$dataset)
bl$seed <- factor(bl$seed)
# Sum-to-zero coding on dataset so the raw method coefficients read as the
# effect for an average dataset (see fit_sweep.R). emmeans results are
# identical under either coding.
contrasts(bl$dataset) <- contr.sum(nlevels(bl$dataset))

fit_add <- glmmTMB(acc ~ method + dataset + (1 | seed),
                   family = beta_family(link = "logit"), data = bl)
fit_bl <- glmmTMB(acc ~ method * dataset + (1 | seed),
                  family = beta_family(link = "logit"), data = bl)
report(fit_bl, "BASELINES  acc ~ method * dataset + (1|seed), sleep as reference")

lrt <- anova(fit_add, fit_bl)
cat("\n--- method x dataset interaction (LRT against the additive model) ---\n")
print(lrt)

write.csv(tidy_fit(fit_bl, "baselines"), file.path(OUT, "baselines_coefs.csv"),
          row.names = FALSE)

# =============================================================================
# Estimated marginal means and contrasts (emmeans)
# =============================================================================
emm_md <- emmeans(fit_bl, ~ method * dataset)   # all 20 cells
emm_m <- emmeans(fit_bl, ~ method)              # averaged over datasets

s_md <- summary(emm_md, type = "response", infer = c(TRUE, FALSE))
pred_ds <- data.frame(dataset = as.character(s_md$dataset),
                      method = as.character(s_md$method),
                      fit = s_md$response, lo = s_md$asymp.LCL, hi = s_md$asymp.UCL)
write.csv(pred_ds[order(pred_ds$dataset, pred_ds$method), ],
          file.path(OUT, "baselines_predictions_by_dataset.csv"), row.names = FALSE)

s_m <- summary(emm_m, type = "response", infer = c(TRUE, FALSE))
write.csv(data.frame(method = as.character(s_m$method), fit = s_m$response,
                     lo = s_m$asymp.LCL, hi = s_m$asymp.UCL),
          file.path(OUT, "baselines_predictions.csv"), row.names = FALSE)

# 1. Each method vs sleep, averaged over datasets (main text).
# 2. Each method vs sleep within one dataset (appendix).
# Unadjusted Wald tests, as in the sweep.
ctr <- as.data.frame(contrast(emm_m, "trt.vs.ctrl1", adjust = "none"))
ctr_ds <- as.data.frame(contrast(emm_md, "trt.vs.ctrl1", by = "dataset",
                                 adjust = "none"))
method_of <- function(x) sub(" *-.*$", "", as.character(x))
ctr$method <- method_of(ctr$contrast)
ctr_ds$method <- method_of(ctr_ds$contrast)
ctr_ds <- ctr_ds[order(ctr_ds$dataset, ctr_ds$method),
                 c("dataset", "method", "estimate", "SE", "z.ratio", "p.value")]
names(ctr_ds) <- c("dataset", "method", "estimate", "std_error", "z", "p")
write.csv(ctr_ds, file.path(OUT, "baselines_contrasts_by_dataset.csv"),
          row.names = FALSE)

# =============================================================================
# Regression table
# =============================================================================
# Label the sleep row with the ratio the sleep arm actually ran at (it comes
# from the sweep, so it is not fixed at 10%).
sleep_pct <- round(100 * max(bl$sleep_ratio[as.character(bl$method) == "sleep"], na.rm = TRUE))
label <- c(sleep = sprintf("Sleep %d\\%%", sleep_pct), none = "No regularization",
           decay = "Weight decay", norm_layer = "Layer normalization",
           norm_neuron = "Synaptic scaling")
order_terms <- c("none", "decay", "norm_layer", "norm_neuron")

stars <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**",
                     ifelse(p < 0.05, "*", "")))

# One block averaged over datasets, then one block per dataset. Each block is
# sleep's accuracy (logit scale) followed by every other method minus sleep.
# The dataset blocks are the simple effects in baselines_contrasts_by_dataset.csv;
# they are what the averaged block can hide (decay is level with sleep on MNIST
# and far below it elsewhere).
datasets <- c(mnist = "MNIST", kmnist = "KMNIST", fmnist = "Fashion-MNIST",
              notmnist = "NotMNIST")

ref_avg <- as.data.frame(summary(emm_m, infer = c(FALSE, TRUE)))
ref_avg <- ref_avg[ref_avg$method == "sleep", ]
ref_ds <- as.data.frame(summary(emm_md, infer = c(FALSE, TRUE)))
ref_ds <- ref_ds[ref_ds$method == "sleep", ]

block_df <- function(block, ref, c) {
    c <- c[match(order_terms, c$method), ]
    rbind(data.frame(block = block, term = "sleep", estimate = ref$emmean,
                     std_error = ref$SE, z = ref$z.ratio, p = ref$p.value),
          data.frame(block = block, term = c$method, estimate = c$estimate,
                     std_error = c[["SE"]] %||% c$std_error,
                     z = c[["z.ratio"]] %||% c$z, p = c[["p.value"]] %||% c$p))
}
tab <- block_df("average", ref_avg, ctr)
for (d in names(datasets)) {
    tab <- rbind(tab, block_df(d, ref_ds[ref_ds$dataset == d, ],
                               ctr_ds[ctr_ds$dataset == d, ]))
}
rownames(tab) <- NULL
tab$sig <- stars(tab$p)

vcl <- glmmTMB::VarCorr(fit_bl)$cond
re <- data.frame(group = names(vcl),
                 variance = sapply(vcl, function(v) as.numeric(v)[1]),
                 row.names = NULL)
re$sd <- sqrt(re$variance)

write.csv(tab, file.path(OUT, "baselines_table.csv"), row.names = FALSE)
write.csv(re, file.path(OUT, "baselines_table_random.csv"), row.names = FALSE)

# --- LaTeX, same layout as sweep_table.tex -----------------------------------
apa <- function(x, d = 3) {
    out <- sprintf(paste0("%.", d, "f"), x)
    out <- sub("^0\\.", ".", out)
    sub("^-0\\.", "-.", out)
}
apa_p <- function(p) ifelse(p < 0.001, "$<.001$", apa(p, 3))
tex_row <- function(lab, r) sprintf("        %s & %s & %s & %s & %s & %s\\\\",
    lab, apa(r$estimate), apa(r$std_error), apa(r$z), apa_p(r$p), r$sig)
small <- function(x, d = 3) ifelse(x < 10^(-d), sprintf("$< .%s1$", strrep("0", d - 1)), apa(x, d))
subhead <- function(txt) sprintf("        \\multicolumn{6}{l}{\\textit{%s}}\\\\", txt)
block_tex <- function(block, head) {
    b <- tab[tab$block == block, ]
    out <- subhead(head)
    for (i in seq_len(nrow(b))) {
        lab <- if (b$term[i] == "sleep") label[["sleep"]] else
            paste0("\\quad ", label[[b$term[i]]])
        out <- c(out, tex_row(lab, b[i, ]))
    }
    out
}
header <- "    Effect  & Estimate & Std. Error & \\textit{z} value & \\textit{p}-value & \\\\"

tex <- c(
    "{",
    "\\setlength{\\tabcolsep}{10pt} % Only affects this table",
    "\\begin{longtable}{lrrrrl}",
    "    \\caption{Mixed effect model with beta regression: sleep against conventional stabilization \\label{tab:betareg_baselines}}\\\\",
    "    \\toprule", header, "    \\midrule", "    \\endfirsthead", "",
    "    \\toprule", header, "    \\midrule", "    \\endhead",
    block_tex("average", "Averaged over datasets")
)
for (d in names(datasets)) {
    tex <- c(tex, "        \\midrule", block_tex(d, datasets[[d]]))
}
pretty_grp <- c(dataset = "Dataset", seed = "Seed")
tex <- c(tex, "        \\midrule",
         "        \\multicolumn{2}{l}{Random effect variance}\\\\",
         "        Groups & Name & Variance & Std.Dev.& &\\\\")
for (i in seq_len(nrow(re))) {
    tex <- c(tex, sprintf("        %s & (Intercept) & %s & %s & &\\\\",
                          pretty_grp[[re$group[i]]], small(re$variance[i]),
                          small(re$sd[i])))
}
tex <- c(tex, "", "        \\bottomrule",
         paste0("    \\multicolumn{6}{p{0.9\\linewidth}}{\\footnotesize Estimates on the ",
                "logit scale. In each block the first row is sleep's accuracy and ",
                "the indented rows are each regularizer minus sleep. ",
                "Model: accuracy $\\sim$ method $\\times$ dataset + (1$|$seed); ",
                "the averaged block is taken over the four datasets. ",
                "Significance levels: * $p<.05$, ** $p<.01$, *** $p<.001$.}\\\\"),
         "\\end{longtable}", "}")
writeLines(tex, file.path(OUT, "baselines_table.tex"))

# --- console ---
cat("\n--- table (logit scale, averaged over datasets) ---\n")
print(tab, digits = 4)
print(re, digits = 4)
cat("\n--- per-dataset contrasts vs sleep ---\n")
print(ctr_ds, digits = 3)
cat("\n--- sign agreement across datasets ---\n")
for (m in order_terms) {
    e <- ctr_ds$estimate[ctr_ds$method == m]
    cat(sprintf("%-12s %s\n", m,
                if (max(e) > 0 && min(e) < 0) "CROSSOVER" else "same sign"))
}
cat("\nwrote baselines_coefs.csv, baselines_predictions{,_by_dataset}.csv,",
    "baselines_contrasts_by_dataset.csv, baselines_table.{csv,tex},",
    "baselines_table_random.csv\n")
