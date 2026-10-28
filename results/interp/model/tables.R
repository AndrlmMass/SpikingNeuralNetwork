# GLMM summary tables for the article, in the house style of the previous paper:
# one row per fixed effect (Est. / z / p / stars), random-effect variances and the
# dispersion in a footer block, significance legend in a caption*.
#
#   phase 1  cbind(k, n-k) ~ cond * dataset + (1|seed)      [linear probe]
#   phase 2  cbind(k, n-k) ~ cond * dataset + (1|seed)      [learned readout | probe]
#
# Phase 2 mirrors contrasts.R exactly: dispersion is allowed to vary by dataset when that
# fits better and converges, and the seed intercept is dropped when the fit is singular.
# Writing the tables from the fitted objects means the LaTeX can never drift from the fit.
#
# Usage (from the repo root):
#   Rscript results/interp/model/tables.R

suppressPackageStartupMessages(library(glmmTMB))
CTRL <- glmmTMBControl(optCtrl = list(iter.max = 5000, eval.max = 5000))
OUT  <- "results/tables"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

d <- read.csv("results/interp/model/model_data.csv", stringsAsFactors = TRUE)
d$seed <- factor(d$seed)
ok <- function(m) !is.null(m) && is.finite(AIC(m)) && m$sdr$pdHess

# ---------- formatting helpers ----------------------------------------------
num <- function(x, dig = 3) {               # .024 / -.024, no leading zero
  s <- formatC(abs(x), format = "f", digits = dig)
  s <- sub("^0\\.", ".", s)
  paste0(ifelse(x < 0, "$-$", ""), s)
}
zed <- function(x) {
  s <- formatC(abs(x), format = "f", digits = 2)
  paste0(ifelse(x < 0, "$-$", ""), s)
}
pval <- function(p) ifelse(p < .001, "$<.001$",
                    paste0("$", sub("^0\\.", ".", formatC(p, format = "f", digits = 3)), "$"))
stars <- function(p) ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", "")))

# Readable names for the reference-coded terms glmmTMB reports.
pretty_term <- function(tm) {
  tm <- gsub("^\\(Intercept\\)$", "intercept", tm)
  # Base/+/- names as in the Methods: signed labels add or remove a component,
  # unsigned ones replace it.
  tm <- gsub("condbase_ori",  "Base",           tm)
  tm <- gsub("condbase_rnd",  "Random",         tm)
  tm <- gsub("condfrozen",    "Frozen",         tm)
  tm <- gsub("condtriplet",   "Triplet",        tm)
  tm <- gsub("condvogels",    "$+$iSTDP",       tm)
  tm <- gsub("condee_off",    "$-$Recurrence",  tm)
  tm <- gsub("condie_off",    "$-$Inhibition",  tm)
  tm <- gsub("condrandom",    "Random",         tm)   # phase 2
  tm <- gsub("condoriented",  "Plastic RF",     tm)   # phase 2 reference
  tm <- gsub("priororiented", "oriented RF",    tm)
  tm <- gsub("datasetfmnist", "FMNIST",         tm)
  tm <- gsub("datasetkmnist", "KMNIST",         tm)
  tm <- gsub("datasetnotmnist", "notMNIST",     tm)
  tm <- gsub("datasetmnist",  "MNIST",          tm)
  gsub(":", "$\\\\times$", tm)
}

# A phi in the tens of thousands means the beta-binomial has collapsed to a binomial;
# print it in scientific form and say so rather than showing six digits of noise.
fmt_phi <- function(x) if (x >= 1e4)
  sprintf("$\\hat{\\phi}=%s\\times 10^{%d}$", formatC(x / 10^floor(log10(x)),
          format = "f", digits = 1), floor(log10(x))) else
  sprintf("$\\hat{\\phi}=%s$", formatC(x, format = "f", digits = if (x >= 100) 0 else 1))

# p is Holm-adjusted over every fixed effect except the intercept, within each model: the
# family is all the coefficients a reader could cite from the table, so no claim made
# from it escapes the correction. p_raw keeps the unadjusted Wald p.
coefs <- function(m) {
  s <- summary(m)$coefficients$cond
  o <- data.frame(term = rownames(s), est = s[, 1], z = s[, 3], p_raw = s[, 4],
                  stringsAsFactors = FALSE)
  o$p <- o$p_raw
  fam <- o$term != "(Intercept)"
  o$p[fam] <- p.adjust(o$p_raw[fam], "holm")
  o
}
# NA when the seed intercept was dropped or estimated at zero (singular): either way the
# data cannot identify it, and the table says so rather than printing .0000.
seed_var <- function(m) {
  v <- VarCorr(m)$cond$seed
  if (is.null(v) || as.numeric(v) < 1e-6) NA else as.numeric(v)
}
phi <- function(m) {
  if (length(fixef(m)$disp) > 1) NA else exp(unname(fixef(m)$disp[1]))
}

# ---------- phase 1: one model ----------------------------------------------
p1 <- droplevels(subset(d, phase == 1 & response == "probe"))
# Referenced to the base configuration, so every configuration coefficient is one change
# from it -- the same reading as the Base/+/- labels in the phase-1 figure.
p1$cond <- relevel(p1$cond, ref = "base_ori")
p1$dataset <- relevel(p1$dataset, ref = "mnist")
m1 <- glmmTMB(cbind(k, n - k) ~ cond * dataset + (1 | seed),
              family = betabinomial(), data = p1, control = CTRL)
if (!ok(m1)) m1 <- glmmTMB(cbind(k, n - k) ~ cond * dataset,
                           family = betabinomial(), data = p1, control = CTRL)

C1 <- coefs(m1)

# Long form, one row per fixed effect, in glmmTMB's coefficient order so that row k is
# beta_k and the Results text can cite it. p and the stars are Holm-adjusted (see coefs).
row1 <- function(i) sprintf("    $\\hat{\\beta}_{%d}$ (%s) & %s & %s & %s & %s\\\\",
  i - 1, pretty_term(C1$term[i]), num(C1$est[i]), zed(C1$z[i]), pval(C1$p[i]), stars(C1$p[i]))
grp <- function(re) which(grepl(re, C1$term))
icf <- grp("^cond[^:]*$"); ids <- grp("^dataset"); iix <- grp(":")

L <- c(
  "\\footnotesize{",
  "\\begin{longtable}{lrrrl}",
  "    \\caption{Experiment 1 beta-binomial GLMM of linear classifier accuracy: fixed-effect estimates",
  "    on the logit scale. The reference cell is \\textit{Base} (oriented receptive fields",
  "    under trace-STDP, with recurrent excitation and fixed lateral inhibition) on MNIST,",
  "    so each \\textit{configuration} row is the effect on MNIST of changing that one",
  "    component and each interaction row is how that effect changes on the named",
  "    dataset.\\label{tab:glmm_phase1}}\\\\",
  "    \\toprule",
  "    \\textbf{Fixed effect} & Est. & \\textit{z} & \\textit{p}$_{\\text{Holm}}$ & \\\\",
  "    \\midrule",
  "    \\endfirsthead",
  "    \\toprule",
  "    Fixed effect & Est. & \\textit{z} & \\textit{p}$_{\\text{Holm}}$ & \\\\",
  "    \\midrule",
  "    \\endhead",
  sprintf("    $\\hat{\\beta}_0$ (intercept) & %s & %s & %s & %s\\\\",
          num(C1$est[1]), zed(C1$z[1]), pval(C1$p[1]), stars(C1$p[1])),
  "    \\multicolumn{5}{l}{\\textit{Configuration (vs.\\ Base)}}\\\\",
  sapply(icf, row1),
  "    \\multicolumn{5}{l}{\\textit{Dataset (vs.\\ MNIST)}}\\\\",
  sapply(ids, row1),
  "    \\multicolumn{5}{l}{\\textit{Configuration $\\times$ dataset}}\\\\",
  sapply(iix, row1),
  "    \\midrule",
  "    \\multicolumn{5}{l}{\\textbf{Random effect variance}}\\\\",
  sprintf("    Seed & \\multicolumn{2}{r}{%s} & &\\\\",
          if (is.na(seed_var(m1))) "---" else num(seed_var(m1), 4)),
  "    \\multicolumn{5}{l}{\\textbf{Dispersion}}\\\\",
  sprintf("    Model & \\multicolumn{2}{r}{%s} & &\\\\", fmt_phi(phi(m1))),
  "    \\bottomrule",
  sprintf(paste0("    \\caption*{$p$-values are Holm-adjusted over the %d fixed effects other than ",
                 "the intercept. Significance: * $p<.05$, ** $p<.01$, *** $p<.001$.%s%s}"),
          sum(C1$term != "(Intercept)"),
          if (is.na(seed_var(m1))) " The seed intercept was singular and was dropped." else "",
          if (phi(m1) >= 1e4) " The estimated dispersion is large, so the fit is effectively binomial." else ""),
  "\\end{longtable}",
  "}")
writeLines(unlist(L), file.path(OUT, "tab_glmm_phase1.tex"))

# ---------- phase 2: two models side by side --------------------------------
fit2 <- function(resp) {
  p2 <- droplevels(subset(d, phase == 2 & response == resp))
  p2$cond <- factor(p2$cond, levels = c("oriented", "random", "frozen"))   # Plastic RF first
  p2$dataset <- relevel(p2$dataset, ref = "mnist")
  m0 <- glmmTMB(cbind(k, n - k) ~ cond * dataset + (1 | seed),
                family = betabinomial(), data = p2, control = CTRL)
  m1 <- tryCatch(glmmTMB(cbind(k, n - k) ~ cond * dataset + (1 | seed),
                         family = betabinomial(), dispformula = ~ dataset,
                         data = p2, control = CTRL), error = function(e) NULL)
  use_var <- ok(m1) && (!ok(m0) || AIC(m1) < AIC(m0))
  m <- if (use_var) m1 else m0
  re <- TRUE
  if (!ok(m)) {
    m <- glmmTMB(cbind(k, n - k) ~ cond * dataset,
                 family = betabinomial(), data = p2, control = CTRL)
    use_var <- FALSE; re <- FALSE
  }
  list(m = m, var_disp = use_var, re = re, n = nrow(p2))
}
FL <- fit2("learned"); FP <- fit2("probe")
CL <- coefs(FL$m); CP <- coefs(FP$m)
stopifnot(identical(CL$term, CP$term))

est_star <- function(e, p) paste0(num(e), if (nzchar(stars(p)))
  paste0("$^{", stars(p), "}$") else "")
row2 <- function(i, lab) sprintf("    %s & %s & %s & %s & %s & %s & %s\\\\",
  lab, est_star(CL$est[i], CL$p[i]), zed(CL$z[i]), pval(CL$p[i]),
       est_star(CP$est[i], CP$p[i]), zed(CP$z[i]), pval(CP$p[i]))

ii  <- grepl(":", CL$term)
ids <- !ii & grepl("^dataset", CL$term)
ipr <- !ii & grepl("^cond", CL$term)
# Which decoders' seed intercepts were singular (dropped, or estimated at zero).
sing <- c(if (!FL$re || is.na(seed_var(FL$m))) "learned readout",
          if (!FP$re || is.na(seed_var(FP$m))) "linear classifier")

# Dispersion per dataset when it was modelled that way; otherwise the single fitted phi,
# shown on the first row and left blank below it rather than repeated four times.
phi_by <- function(m, lv) {
  b <- fixef(m)$disp
  sapply(lv, function(ds) exp(if (ds == lv[1]) b[[1]] else b[[1]] + b[[paste0("dataset", ds)]]))
}
phi_num <- function(x) if (x >= 1e4)          # the value alone; the row label names it
  sprintf("$%s\\times 10^{%d}$", formatC(x / 10^floor(log10(x)), format = "f", digits = 1),
          floor(log10(x))) else formatC(x, format = "f", digits = 0)
disp_cells <- function(F, lv) {
  if (F$var_disp) sapply(unname(phi_by(F$m, lv)), phi_num)
  else c(paste0(phi_num(phi(F$m)), " (shared)"), rep("", length(lv) - 1))
}
LV2 <- c("mnist", setdiff(levels(droplevels(subset(d, phase == 2)$dataset)), "mnist"))
disp_rows <- function() {
  if (!FL$var_disp && !FP$var_disp)
    return(sprintf("    Model & \\multicolumn{3}{c}{%s} & \\multicolumn{3}{c}{%s}\\\\",
                   phi_num(phi(FL$m)), phi_num(phi(FP$m))))
  cl <- disp_cells(FL, LV2); cp <- disp_cells(FP, LV2)
  sprintf("    $\\hat{\\phi}_{\\text{%s}}$ & \\multicolumn{3}{c}{%s} & \\multicolumn{3}{c}{%s}\\\\",
          sapply(LV2, function(x) pretty_term(paste0("dataset", x))), cl, cp)
}
seed_cell <- function(F) if (!F$re || is.na(seed_var(F$m))) "--- (singular)" else
  num(seed_var(F$m), 4)

L2 <- c(
  "\\footnotesize{",
  "\\setlength{\\tabcolsep}{4pt}",
  "\\begin{longtable}{l rrr rrr}",
  "    \\caption{Experiment 2 beta-binomial GLMM of supervised accuracy, by decoder: fixed-effect",
  "    estimates on the logit scale. The reference cell is \\textit{Plastic RF} (receptive",
  "    fields trained with R-Hebb) on MNIST, so \\textit{Random} and \\textit{Frozen} are the effects on",
  "    MNIST of replacing the receptive fields with random connectivity and of disabling",
  "    R-Hebb, and each interaction row is how that effect changes on the named",
  "    dataset.\\label{tab:glmm_phase2}}\\\\",
  "    \\toprule",
  "    & \\multicolumn{3}{c}{\\textit{Learned readout}} & \\multicolumn{3}{c}{\\textit{Linear classifier}}\\\\",
  "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
  "    \\textbf{Fixed effect} & Est. & \\textit{z} & \\textit{p}$_{\\text{Holm}}$ & Est. & \\textit{z} & \\textit{p}$_{\\text{Holm}}$\\\\",
  "    \\midrule",
  "    \\endfirsthead",
  "    \\toprule",
  "    & \\multicolumn{3}{c}{\\textit{Learned readout}} & \\multicolumn{3}{c}{\\textit{Linear classifier}}\\\\",
  "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
  "    Fixed effect & Est. & \\textit{z} & \\textit{p}$_{\\text{Holm}}$ & Est. & \\textit{z} & \\textit{p}$_{\\text{Holm}}$\\\\",
  "    \\midrule",
  "    \\endhead",
  row2(1, "$\\hat{\\beta}_0$ (intercept)"),
  "    \\multicolumn{7}{l}{\\textit{Main effects}}\\\\",
  sapply(which(ipr), function(i)
    row2(i, sprintf("$\\hat{\\beta}_{%d}$ (%s)", i - 1, pretty_term(CL$term[i])))),
  sapply(which(ids), function(i)
    row2(i, sprintf("$\\hat{\\beta}_{%d}$ (%s)", i - 1, pretty_term(CL$term[i])))),
  "    \\multicolumn{7}{l}{\\textit{Two-way interactions}}\\\\",
  sapply(which(ii), function(i)
    row2(i, sprintf("$\\hat{\\beta}_{%d}$ (%s)", i - 1, pretty_term(CL$term[i])))),
  "    \\midrule",
  "    \\multicolumn{7}{l}{\\textbf{Random effect variance}}\\\\",
  sprintf("    Seed & \\multicolumn{3}{c}{%s} & \\multicolumn{3}{c}{%s}\\\\",
          seed_cell(FL), seed_cell(FP)),
  "    \\multicolumn{7}{l}{\\textbf{Dispersion}}\\\\",
  disp_rows(),
  "    \\bottomrule",
  sprintf(paste0("    \\caption*{$p$-values are Holm-adjusted over the fixed effects other than the ",
                 "intercept, within each decoder's model. ",
                 "Significance: * $p<.05$, ** $p<.01$, *** $p<.001$. Reference cell ",
                 "= \\textit{Plastic RF} on MNIST. %s%s}"),
          if (FL$var_disp) "The learned readout is fitted with one dispersion per dataset." else "",
          if (length(sing)) sprintf(" The seed intercept was singular for the %s and was dropped.",
                                    paste(sing, collapse = " and ")) else ""),
  "\\end{longtable}",
  "\\setlength{\\tabcolsep}{6pt}}")
writeLines(unlist(L2), file.path(OUT, "tab_glmm_phase2.tex"))

cat("wrote", file.path(OUT, "tab_glmm_phase1.tex"), "and tab_glmm_phase2.tex\n")
