# =============================================================================
# SLEEP-COMPONENT ABLATION -- 2^4 factorial, 4 datasets, 5 seeds
# =============================================================================
# Each sleep component (downscaling, membrane noise, sleep-phase STDP, input
# suppression) is switched on or off. One beta regression (logit link, one
# shared precision parameter):
#
#   acc ~ downscale * noise * stdp * suppress
#         + (1 | dataset) + (1 | seed) + (1 | dataset:seed)
#
# The table reports the raw coefficients, with all components off as the
# reference level.
#
# Outputs (results/glmm/): ablation_table.tex, ablation_ds_coefs.csv
# Figure inputs are written by ablation_figure_data.R.
#
# Usage (from the repo root):  Rscript src/glmm/fit_ablation_datasets.R
# =============================================================================

source("src/glmm/common.R")
ab <- read_ablation()$runs

fit <- glmmTMB(acc ~ downscale * noise * stdp * suppress
                   + (1 | dataset) + (1 | seed) + (1 | dataset:seed),
               family = beta_family(link = "logit"), data = ab)
report(fit, "ABLATION  acc ~ downscale * noise * stdp * suppress + (1|dataset) + (1|seed) + (1|dataset:seed)")
coefs <- tidy_fit(fit, "ablation")
write.csv(coefs, file.path(OUT, "ablation_ds_coefs.csv"), row.names = FALSE)

# --- Table --------------------------------------------------------------------
# Row label from a coefficient name, e.g. "downscaleon:stdpon" -> "Downscaling x STDP".
# Single components get their full name; interactions use short names so the
# rows fit the page width.
LONG <- c(downscale = "Downscaling", stdp = "STDP", noise = "Membrane noise",
          suppress = "Input suppression")
SHORT <- c(downscale = "Downscaling", stdp = "STDP", noise = "noise", suppress = "suppression")
term_label <- function(term) {
    if (term == "(Intercept)") return("Intercept (all off)")
    parts <- sub("on$", "", strsplit(term, ":")[[1]])
    if (length(parts) == 1) return(LONG[[parts]])
    if (length(parts) == 4) return("All four components")
    lab <- SHORT[parts]
    lab[1] <- paste0(toupper(substr(lab[1], 1, 1)), substring(lab[1], 2))
    paste(lab, collapse = " $\\times$ ")
}
stars <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "")))
coef_rows <- function(cf) sprintf("        %s & %s & %s & %s & %s & %s\\\\",
    sapply(cf$term, term_label), apa(cf$estimate), apa(cf$std_error), apa(cf$z),
    apa_p(cf$p), stars(cf$p))

# The table shows the intercept and the single-component coefficients only;
# the interaction terms are in ablation_ds_coefs.csv.
order <- ifelse(coefs$term == "(Intercept)", 0, lengths(strsplit(coefs$term, ":")))
rows <- coef_rows(coefs[order <= 1, ])

header <- "    Effect  & Estimate & Std. Error & \\textit{z} value & \\textit{p}-value & \\\\"
note <- paste0("Coefficients on the logit scale, with all components off as the reference: ",
               "the intercept is the all-off condition and each estimate is the change in ",
               "log odds when only that component is switched on. Interaction terms are ",
               "estimated but not shown; none is significant. ",
               "Model: accuracy $\\sim$ downscaling $\\times$ STDP $\\times$ membrane noise ",
               "$\\times$ input suppression + (1$|$dataset) + (1$|$seed) + (1$|$dataset:seed), ",
               "with a single precision parameter; estimates are for a typical dataset and ",
               "seed. Significance levels: * $p<.05$, ** $p<.01$, *** $p<.001$.")
writeLines(c(
    "{", "\\setlength{\\tabcolsep}{10pt} % Only affects this table",
    "\\begin{longtable}{lrrrrl}",
    "    \\caption{Mixed effect model with beta regression: effect of each sleep component \\label{tab:betareg_ablation}}\\\\",
    "    \\toprule", header, "    \\midrule", "    \\endfirsthead", "",
    "    \\toprule", header, "    \\midrule", "    \\endhead",
    rows, re_tex(random_effects(fit)), "", "        \\bottomrule",
    sprintf("    \\multicolumn{6}{p{0.9\\linewidth}}{\\footnotesize %s}\\\\", note),
    "\\end{longtable}", "}"), file.path(OUT, "ablation_table.tex"))

cat("\nwrote ablation_table.tex, ablation_ds_coefs.csv\n")
