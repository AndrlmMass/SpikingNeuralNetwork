# =============================================================================
# Inputs for the two ablation figures (plot_glmms_bw.py)
# =============================================================================
# Uses the ablation interaction model (see fit_ablation_datasets.R) and nothing else:
# for each component, accuracy predicted from the coefficients with that
# component off (the intercept: all components off) and on (intercept + that
# component's coefficient). No averaging over conditions.
#
#   ablation_ds_marginal.csv           per dataset (that dataset's random
#                                      intercept, seed at zero), 95% CI
#   ablation_pop_marginal.csv          typical dataset and seed (random effects
#                                      at zero), 95% CI
#   ablation_ds_nosleep_reference.csv  mean accuracy of the no-sleep runs
#
# Usage (from the repo root):  Rscript src/glmm/ablation_figure_data.R
# =============================================================================

source("src/glmm/common.R")

d <- read_ablation()
ab <- d$runs
comps <- ABLATION_COMPONENTS
fit <- glmmTMB(acc ~ downscale * noise * stdp * suppress
                   + (1 | dataset) + (1 | seed) + (1 | dataset:seed),
               family = beta_family(link = "logit"), data = ab)

# One row per (component, state): all components off, except `component`
# when state is "on".
cells <- do.call(rbind, lapply(comps, function(v) {
    g <- data.frame(component = v, state = c("off", "on"))
    for (w in comps) g[[w]] <- factor(if (w == v) g$state else "off", levels = c("off", "on"))
    g
}))

by_ds <- do.call(rbind, lapply(levels(ab$dataset), function(ds)
    cbind(dataset = ds, cells)))
by_ds$dataset <- factor(by_ds$dataset, levels = levels(ab$dataset))
by_ds <- cond_pred(fit, by_ds, ab$seed)
write.csv(by_ds[, c("dataset", "component", "state", "fit", "lo", "hi")],
          file.path(OUT, "ablation_ds_marginal.csv"), row.names = FALSE)

pop <- pop_pred(fit, cells)
write.csv(pop[, c("component", "state", "fit", "lo", "hi")],
          file.path(OUT, "ablation_pop_marginal.csv"), row.names = FALSE)
print(pop[, c("component", "state", "fit", "lo", "hi")], digits = 3)

refs <- aggregate(test_accuracy ~ dataset, data = d$nosleep, FUN = mean)
names(refs)[2] <- "mean_acc"
refs$n <- aggregate(test_accuracy ~ dataset, data = d$nosleep, FUN = length)$test_accuracy
write.csv(refs, file.path(OUT, "ablation_ds_nosleep_reference.csv"), row.names = FALSE)

cat("wrote ablation_ds_marginal.csv, ablation_pop_marginal.csv, ablation_ds_nosleep_reference.csv\n")
