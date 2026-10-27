# Phase 1 planned contrasts for the Results text, from the same cond * dataset model as
# tables.R. Each contrast is a linear combination of the fixed effects:
#   * Wald z and p on the logit scale (the test),
#   * the difference in percentage points from 4000 draws of the fixed effects (the size),
#   * Holm-adjusted p within each family of contrasts.
# "overall" rows average the logit contrast over the four datasets with equal weight,
# which is the single test for "A beats B in general".
#
# Usage (from the repo root):
#   Rscript results/interp/model/phase1_contrasts.R

suppressPackageStartupMessages({library(glmmTMB); library(MASS)})
set.seed(3)
CTRL <- glmmTMBControl(optCtrl = list(iter.max = 5000, eval.max = 5000))
ok <- function(m) !is.null(m) && is.finite(AIC(m)) && m$sdr$pdHess

d <- read.csv("results/interp/model/model_data.csv", stringsAsFactors = TRUE)
d$seed <- factor(d$seed)
p1 <- droplevels(subset(d, phase == 1 & response == "probe"))
p1$cond <- relevel(p1$cond, ref = "base_ori")
p1$dataset <- relevel(p1$dataset, ref = "mnist")
m <- glmmTMB(cbind(k, n - k) ~ cond * dataset + (1 | seed),
             family = betabinomial(), data = p1, control = CTRL)
if (!ok(m)) m <- glmmTMB(cbind(k, n - k) ~ cond * dataset,
                         family = betabinomial(), data = p1, control = CTRL)

b <- fixef(m)$cond; V <- vcov(m)$cond
grid <- expand.grid(cond = levels(p1$cond), dataset = levels(p1$dataset))
X <- model.matrix(~ cond * dataset, data = grid)
stopifnot(identical(colnames(X), names(b)))
B <- mvrnorm(4000, b, V)
ETA <- B %*% t(X)                                  # draws x cells, logit scale
row_of <- function(cd, ds) which(grid$cond == cd & grid$dataset == ds)
DS <- levels(p1$dataset)

# a contrast = weights over grid cells
contrast <- function(label, family, w) {
  est <- sum(w * (X %*% b)); se <- sqrt(drop(t(w) %*% X %*% V %*% t(X) %*% w))
  pp  <- plogis(ETA) %*% w * 100
  data.frame(family = family, contrast = label, logit = est, z = est / se,
             p = 2 * pnorm(-abs(est / se)), pp = mean(pp),
             lo = quantile(pp, .025), hi = quantile(pp, .975), row.names = NULL)
}
pair <- function(a, r, family) {
  out <- lapply(DS, function(ds) {
    w <- numeric(nrow(grid)); w[row_of(a, ds)] <- 1; w[row_of(r, ds)] <- -1
    contrast(sprintf("%s - %s | %s", a, r, ds), family, w)
  })
  w <- numeric(nrow(grid))
  for (ds in DS) { w[row_of(a, ds)] <- 1 / length(DS); w[row_of(r, ds)] <- -1 / length(DS) }
  rbind(do.call(rbind, out), contrast(sprintf("%s - %s | overall", a, r), family, w))
}

res <- rbind(
  # 1. connectivity: the only random cell is trace-STDP, so the RF effect is base_ori vs base_rnd
  pair("base_ori", "base_rnd", "RF vs random (trace-STDP)"),
  # 2. every plastic configuration against frozen weights
  do.call(rbind, lapply(setdiff(levels(p1$cond), "frozen"),
                        function(c) pair(c, "frozen", "vs frozen"))),
  # 3. each single change against the base configuration (the model's reference level)
  do.call(rbind, lapply(c("triplet", "vogels", "frozen", "ee_off", "ie_off", "base_rnd"),
                        function(c) pair(c, "base_ori", "vs base")))
)
# Holm within each family, per-dataset rows only (the overall rows are single planned tests)
res$p_holm <- res$p
for (f in unique(res$family)) {
  i <- res$family == f & !grepl("overall", res$contrast)
  res$p_holm[i] <- p.adjust(res$p[i], "holm")
}
write.csv(res, "results/interp/model/phase1_contrasts.csv", row.names = FALSE)

# For the averaged phase-1 figure (plot_phase1_average.py): each configuration's predicted
# accuracy, averaged over the four datasets with equal weight, one value per joint
# fixed-effect draw. Uses no further random numbers, so the table above is unchanged.
avg <- do.call(rbind, lapply(levels(p1$cond), function(cd) {
  w <- numeric(nrow(grid))
  for (ds in DS) w[row_of(cd, ds)] <- 1 / length(DS)
  data.frame(cond = cd, acc = drop(plogis(ETA) %*% w))
}))
write.csv(avg, "results/interp/model/phase1_average_draws.csv", row.names = FALSE)

fmt <- function(r) sprintf("%-44s %+6.2f pp [%+6.2f, %+6.2f]  z=%7.2f  p=%.2g  p_holm=%.2g",
                           r$contrast, r$pp, r$lo, r$hi, r$z, r$p, r$p_holm)
for (f in unique(res$family)) {
  cat("\n==", f, "==\n")
  s <- res[res$family == f, ]
  for (i in seq_len(nrow(s))) cat(fmt(s[i, ]), "\n")
}
cat("\nwrote results/interp/model/phase1_contrasts.csv\n")
