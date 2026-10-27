# oriented - random per dataset, phase 2.
# Dispersion is allowed to vary by dataset where that fits better AND converges; the
# seed spread differs ~15x across datasets, and forcing one phi understates uncertainty
# exactly where the paper makes its negative claims.
suppressPackageStartupMessages({library(glmmTMB); library(MASS)})
set.seed(2)
CTRL <- glmmTMBControl(optCtrl = list(iter.max = 5000, eval.max = 5000))
d <- read.csv("results/interp/model/model_data.csv", stringsAsFactors = TRUE); d$seed <- factor(d$seed)
ok <- function(m) !is.null(m) && is.finite(AIC(m)) && m$sdr$pdHess

res <- list()
for (resp in c("learned", "probe")) {
  p2 <- droplevels(subset(d, phase == 2 & response == resp)); p2$prior <- relevel(p2$prior, ref = "random")
  m0 <- glmmTMB(cbind(k,n-k) ~ prior*dataset + (1|seed), family=betabinomial(), data=p2, control=CTRL)
  m1 <- tryCatch(glmmTMB(cbind(k,n-k) ~ prior*dataset + (1|seed), family=betabinomial(),
                         dispformula = ~ dataset, data=p2, control=CTRL), error=function(e) NULL)
  use_var <- ok(m1) && (!ok(m0) || AIC(m1) < AIC(m0))
  m <- if (use_var) m1 else m0
  re <- "seed RE"
  if (!ok(m)) {
    # Singular fit: seed variance ~ 0 drives phi -> Inf (the probe after the 09-16 rerun).
    # Drop the seed intercept, which the data cannot identify, and keep the dispersion.
    m <- glmmTMB(cbind(k,n-k) ~ prior*dataset, family=betabinomial(), data=p2, control=CTRL)
    use_var <- FALSE; re <- "no seed RE (singular)"
  }
  cat(sprintf("\n== phase 2 / %s ==  dispersion: %s   %s   AIC %.1f\n", resp,
              if (use_var) "phi ~ dataset" else "constant phi", re, AIC(m)))
  b <- fixef(m)$cond; V <- vcov(m)$cond; B <- mvrnorm(4000, b, V)
  nd <- unique(p2[,c("prior","dataset")]); X <- model.matrix(~ prior*dataset, data=nd)
  ETA <- B %*% t(X)
  # Test = Wald z on the logit-scale contrast (as in phase1_contrasts.R); size = pp from the
  # draws. The four datasets of one decoder form one Holm family, applied below.
  rr <- list()
  for (ds in levels(p2$dataset)) {
    io <- which(nd$prior=="oriented" & nd$dataset==ds); ir <- which(nd$prior=="random" & nd$dataset==ds)
    dd <- plogis(ETA[,io]) - plogis(ETA[,ir]); p_sim <- 2*min(mean(dd<=0), mean(dd>=0))
    w <- X[io,] - X[ir,]; lg <- sum(w * b); se <- sqrt(drop(t(w) %*% V %*% w))
    rr[[ds]] <- data.frame(response=resp, dataset=ds, est=mean(dd),
        lo=unname(quantile(dd,.025)), hi=unname(quantile(dd,.975)),
        logit=lg, z=lg/se, p=2*pnorm(-abs(lg/se)), p_sim=p_sim,
        disp=if (use_var) "phi~dataset" else "constant", re=re)
  }
  rr <- do.call(rbind, rr); rr$p_holm <- p.adjust(rr$p, "holm")
  cat(sprintf("  %-9s %8s %20s %8s %10s %10s\n","dataset","delta pp","95%% CI (pp)","z","p","p_holm"))
  for (i in seq_len(nrow(rr))) cat(sprintf("  %-9s %+8.2f   [%+6.2f, %+6.2f] %8.2f %10.2g %10.2g\n",
      rr$dataset[i], rr$est[i]*100, rr$lo[i]*100, rr$hi[i]*100, rr$z[i], rr$p[i], rr$p_holm[i]))
  res[[resp]] <- rr
}
write.csv(do.call(rbind,res), "results/interp/model/contrasts.csv", row.names=FALSE)
cat("\nwrote results/interp/model/contrasts.csv\n")
