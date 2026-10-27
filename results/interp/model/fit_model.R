# Beta-binomial mixed models for the RF-article sweeps.
#
# Response = successes / trials on the held-out test set, so the differing denominators
# (2500 notMNIST .. 25000 SVHN) weight the fit correctly; the beta-binomial dispersion
# absorbs seed-to-seed variation a plain binomial would understate.
#
# Phase 1: `cond` is the 7-level ablation factor, referenced to `base_ori` (trace-STDP with
#   E->E recurrence and I->E inhibition, oriented prior), so every other level is one change
#   from it. The prior is aliased with cond (it varies only in base_ori vs base_rnd) so it
#   cannot be a separate term; it is recovered as that contrast.
# Phase 2: prior is crossed with dataset -> proper main effect plus a prior x dataset
#   interaction, which is the formal test of the conditional-advantage claim.

suppressPackageStartupMessages(library(glmmTMB))
CTRL <- glmmTMBControl(optCtrl = list(iter.max = 5000, eval.max = 5000))

d <- read.csv("results/interp/model/model_data.csv", stringsAsFactors = TRUE)
d$seed <- factor(d$seed)

lrt <- function(m_small, m_big, label) {
  ll0 <- as.numeric(logLik(m_small)); ll1 <- as.numeric(logLik(m_big))
  df  <- attr(logLik(m_big), "df") - attr(logLik(m_small), "df")
  chi <- 2 * (ll1 - ll0); p <- pchisq(chi, df, lower.tail = FALSE)
  cat(sprintf("\n  LRT %s: chisq = %.1f, df = %d, p = %s\n", label, chi, df,
              format.pval(p, digits = 3, eps = 1e-300)))
}

run <- function(dat, big, small, label, grp) {
  cat("\n", strrep("=", 76), "\n", label, "\n", strrep("=", 76), "\n", sep = "")
  mB <- glmmTMB(big,   family = betabinomial(), data = dat, control = CTRL)
  mS <- glmmTMB(small, family = betabinomial(), data = dat, control = CTRL)
  if (!mB$sdr$pdHess || !mS$sdr$pdHess) {
    # singular seed intercept (variance ~ 0): refit both without it
    cat("  [singular fit: dropping (1|seed)]\n")
    big <- update(big, . ~ . - (1|seed)); small <- update(small, . ~ . - (1|seed))
    mB <- glmmTMB(big,   family = betabinomial(), data = dat, control = CTRL)
    mS <- glmmTMB(small, family = betabinomial(), data = dat, control = CTRL)
  }
  sd_seed <- if (is.null(VarCorr(mB)$cond$seed)) NA else signif(sqrt(as.numeric(VarCorr(mB)$cond$seed)), 3)
  cat("  converged:", mB$fit$convergence == 0, " AIC:", round(AIC(mB), 1),
      " seed SD:", sd_seed, "\n")
  lrt(mS, mB, paste(grp, "x dataset interaction"))
  mB
}

p1 <- droplevels(subset(d, phase == 1 & response == "probe"))
p1$cond <- relevel(p1$cond, ref = "base_ori")
m1 <- run(p1, cbind(k, n-k) ~ cond * dataset + (1|seed),
              cbind(k, n-k) ~ cond + dataset + (1|seed),
          "PHASE 1  linear probe  ~ cond * dataset", "cond")

m2 <- list()
for (resp in c("probe", "learned")) {
  p2 <- droplevels(subset(d, phase == 2 & response == resp))
  p2$prior <- relevel(p2$prior, ref = "random")
  m2[[resp]] <- run(p2, cbind(k, n-k) ~ prior * dataset + (1|seed),
                        cbind(k, n-k) ~ prior + dataset + (1|seed),
                    paste0("PHASE 2  ", resp, "  ~ prior * dataset"), "prior")
}

# ---- population-level predictions (random effects zeroed) -------------------
pc <- function(m, dat, grp, phase, resp) {
  nd <- unique(dat[, c(grp, "dataset", "n")]); nd$seed <- dat$seed[1]
  p  <- predict(m, newdata = nd, type = "link", se.fit = TRUE, re.form = NA)
  o  <- data.frame(group = nd[[grp]], dataset = nd$dataset, phase = phase, response = resp,
                   fit = plogis(p$fit), lo = plogis(p$fit - 1.96*p$se.fit),
                   hi  = plogis(p$fit + 1.96*p$se.fit), eta = p$fit, se_link = p$se.fit)
  o[order(o$dataset, o$group), ]
}
pred <- rbind(
  pc(m1, p1, "cond", 1, "probe"),
  pc(m2[["probe"]],   droplevels(subset(d, phase==2 & response=="probe")),   "prior", 2, "probe"),
  pc(m2[["learned"]], droplevels(subset(d, phase==2 & response=="learned")), "prior", 2, "learned"))
write.csv(pred, "results/interp/model/predictions.csv", row.names = FALSE)

# ---- phase-2 oriented-vs-random contrast per dataset ------------------------
cat("\n", strrep("=", 76), "\n  PHASE 2 oriented - random, per dataset (pp, on the response scale)\n",
    strrep("=", 76), "\n", sep = "")
for (resp in c("probe","learned")) {
  s <- subset(pred, phase == 2 & response == resp)
  for (ds in levels(droplevels(s$dataset))) {
    a <- s$fit[s$dataset==ds & s$group=="oriented"]; b <- s$fit[s$dataset==ds & s$group=="random"]
    cat(sprintf("  %-8s %-8s oriented %.3f  random %.3f  delta %+6.2f pp\n", resp, ds, a, b, (a-b)*100))
  }
}
cat("\nwrote results/interp/model/predictions.csv (", nrow(pred), "cells )\n")
