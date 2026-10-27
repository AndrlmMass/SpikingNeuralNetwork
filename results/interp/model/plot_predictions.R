# Draw from the fitted models' predictive uncertainty so the boxplots summarise a real
# distribution rather than 5 seeds. eta ~ N(eta_hat, se) on the link scale, back-transformed.
set.seed(1)
p <- read.csv("results/interp/model/predictions.csv", stringsAsFactors = FALSE)
B <- 2000
draws <- do.call(rbind, lapply(seq_len(nrow(p)), function(i) {
  data.frame(group = p$group[i], dataset = p$dataset[i], phase = p$phase[i],
             response = p$response[i],
             value = plogis(rnorm(B, p$eta[i], p$se_link[i])))
}))
write.csv(draws, "results/interp/model/prediction_draws.csv", row.names = FALSE)
cat("wrote", nrow(draws), "draws\n")
