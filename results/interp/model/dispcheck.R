# Does dispersion vary by dataset? Fashion's seed spread is ~15x MNIST's, so a single
# phi would understate uncertainty exactly where the paper makes a negative claim.
suppressPackageStartupMessages({library(glmmTMB); library(MASS)})
set.seed(3)
CTRL <- glmmTMBControl(optCtrl = list(iter.max = 5000, eval.max = 5000))
d <- read.csv("results/interp/model/model_data.csv", stringsAsFactors = TRUE); d$seed <- factor(d$seed)

for (resp in c("learned","probe")) {
  p2 <- droplevels(subset(d, phase == 2 & response == resp)); p2$prior <- relevel(p2$prior, ref="random")
  m0 <- glmmTMB(cbind(k,n-k) ~ prior*dataset + (1|seed), family=betabinomial(), data=p2, control=CTRL)
  m1 <- glmmTMB(cbind(k,n-k) ~ prior*dataset + (1|seed), family=betabinomial(),
                dispformula = ~ dataset, data=p2, control=CTRL)
  ll <- 2*(as.numeric(logLik(m1))-as.numeric(logLik(m0)))
  df <- attr(logLik(m1),"df")-attr(logLik(m0),"df")
  cat(sprintf("\n== %s ==\n  constant phi  AIC %.1f\n  phi ~ dataset AIC %.1f\n  LRT chisq=%.1f df=%d p=%s\n",
      resp, AIC(m0), AIC(m1), ll, df, format.pval(pchisq(ll,df,lower.tail=FALSE),digits=3)))
  best <- if (AIC(m1) < AIC(m0)) m1 else m0
  cat("  -> using:", if (identical(best,m1)) "phi ~ dataset" else "constant phi", "\n")
  b <- fixef(best)$cond; V <- vcov(best)$cond; B <- mvrnorm(4000,b,V)
  nd <- unique(p2[,c("prior","dataset")]); X <- model.matrix(~ prior*dataset, data=nd)
  ETA <- B %*% t(X)
  cat(sprintf("  %-9s %8s %20s %8s\n","dataset","delta pp","95% CI (pp)","p"))
  for (ds in levels(p2$dataset)) {
    io <- which(nd$prior=="oriented" & nd$dataset==ds); ir <- which(nd$prior=="random" & nd$dataset==ds)
    dd <- plogis(ETA[,io])-plogis(ETA[,ir])
    p  <- 2*min(mean(dd<=0), mean(dd>=0))
    cat(sprintf("  %-9s %+8.2f   [%+6.2f, %+6.2f] %8.4f%s\n", ds, mean(dd)*100,
        quantile(dd,.025)*100, quantile(dd,.975)*100, p, ifelse(p<.05,"","   <- n.s.")))
  }
}
