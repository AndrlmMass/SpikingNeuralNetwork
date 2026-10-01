# =============================================================================
# ABLATION — marginal contribution of each component
# =============================================================================
# Same model as fit_ablation.R (Beta GLMM, 2^4 factorial, random seed intercept),
# summarised differently: for each component, the predicted accuracy averaged
# over all 8 settings of the other three components, with the component on and
# off. The average is taken on the accuracy scale (regrid), not the logit scale,
# so "how much accuracy changes" means exactly that.
#
# Usage (from the repo root):
#   Rscript src/glmm/fit_ablation_marginal.R
#
# Output: results/glmm/ablation_marginal.csv
#   one row per component: mean_off, mean_on, diff (on - off) with delta-method
#   95% CI, and grand_mean (average over all 16 cells)
# =============================================================================

source("src/glmm/common.R")
suppressPackageStartupMessages(library(emmeans))

ab <- read.csv(file.path(REPO, "results", "ablation", "ablation_summary.csv"))
ab <- ab[!is.na(ab$test_accuracy) & ab$code != "none", ]
ab$acc <- squeeze(ab$test_accuracy, 700)
comps <- c("downscale", "noise", "stdp", "suppress")
for (v in comps) {
    ab[[v]] <- factor(ifelse(tolower(as.character(ab[[v]])) %in% c("true", "1"),
                             "on", "off"), levels = c("off", "on"))
}
ab$seed <- factor(ab$seed)

fit <- glmmTMB(acc ~ downscale * noise * stdp * suppress + (1 | seed),
               family = beta_family(link = "logit"), data = ab)

reg <- regrid(emmeans(fit, ~ downscale * noise * stdp * suppress),
              transform = "response")
grand <- summary(emmeans(reg, ~ 1))$response

rows <- lapply(comps, function(v) {
    em <- emmeans(reg, as.formula(paste("~", v)))
    m <- as.data.frame(summary(em))
    d <- as.data.frame(confint(pairs(em, reverse = TRUE)))  # on - off
    data.frame(component = v,
               mean_off = m$response[m[[v]] == "off"],
               mean_on = m$response[m[[v]] == "on"],
               diff = d$estimate, diff_lo = d$asymp.LCL, diff_hi = d$asymp.UCL,
               grand_mean = grand)
})
out <- do.call(rbind, rows)
print(out)
write.csv(out, file.path(OUT, "ablation_marginal.csv"), row.names = FALSE)
cat("\nwrote ablation_marginal.csv\n")

# -----------------------------------------------------------------------------
# Single-path effects, the asymmetric companion to the averages above.
#   add:  component on, others off  minus  everything off
#   drop: everything on  minus  component off, others on
# These are differences between two fitted cells, so they carry the
# interaction that the averaged contrast hides.
# -----------------------------------------------------------------------------
em16 <- emmeans(reg, ~ downscale * noise * stdp * suppress)
g <- as.data.frame(em16)
idx <- function(...) {
    on <- c(...)
    which(Reduce(`&`, lapply(comps, function(v) (g[[v]] == "on") == (v %in% on))))
}
coef <- function(plus, minus) {
    k <- numeric(nrow(g)); k[idx(plus)] <- 1; k[idx(minus)] <- -1; k
}
ctr <- list()
for (v in comps) {
    ctr[[paste0("add_", v)]] <- coef(v, character(0))
    # accuracy change from removing v: cell(all but v) - cell(all on)
    ctr[[paste0("drop_", v)]] <- coef(setdiff(comps, v), comps)
}
paths <- as.data.frame(confint(contrast(em16, ctr)))
paths$kind <- sub("_.*", "", paths$contrast)
paths$component <- sub("^[a-z]+_", "", paths$contrast)
paths <- paths[, c("kind", "component", "estimate", "asymp.LCL", "asymp.UCL")]
names(paths) <- c("kind", "component", "est", "lo", "hi")
print(paths)
write.csv(paths, file.path(OUT, "ablation_paths.csv"), row.names = FALSE)
cat("wrote ablation_paths.csv\n")
