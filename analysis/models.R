# Task 6 — statistical analysis of the MesoNet reproduction (run from analysis/).
#
# Inputs : data/predictions.csv (regenerate with `python scripts/export_for_r.py` from repo root)
# Outputs: output/*.csv consumed by report.qmd
#
# The video-clustered GLMM is the rigorous headline analysis: frames from the same source video
# are strongly dependent, so per-frame tests that ignore clustering overstate certainty.

library(readr)
library(dplyr)
library(glmmTMB)
library(broom.mixed)
library(emmeans)
library(pROC)

set.seed(42)
dir.create("output", showWarnings = FALSE)

message("Reading predictions ...")
preds <- read_csv("data/predictions.csv", show_col_types = FALSE)

# ---- in-domain frame-level data for the GLMM ---------------------------------------------
in_domain <- preds |>
  filter(train_source == eval_dataset) |>
  mutate(
    correct  = as.integer((prob_fake > 0.5) == (label == 1)),
    model    = relevel(factor(model), ref = "meso_inception4"),
    method   = relevel(factor(method), ref = "Face2Face"),
    video_id = factor(video_id),
    seed     = factor(seed)
  )
stopifnot(nrow(in_domain) == 12 * 5600)

message("Fitting GLMM: correct ~ model * method + (1|video_id) + (1|seed) ...")
m <- glmmTMB(correct ~ model * method + (1 | video_id) + (1 | seed),
             family = binomial, data = in_domain)

fixed <- tidy(m, effects = "fixed", exponentiate = TRUE, conf.int = TRUE) |>
  select(term, estimate, conf.low, conf.high, p.value)
write_csv(fixed, "output/glmm_fixed_effects.csv")

vc <- VarCorr(m)$cond
ranef_sd <- tibble(
  group = c("video_id", "seed"),
  sd    = c(attr(vc$video_id, "stddev"), attr(vc$seed, "stddev"))
)
write_csv(ranef_sd, "output/glmm_ranef_sd.csv")

message("emmeans per-cell accuracies and model contrasts ...")
emm <- emmeans(m, ~ model | method, type = "response")
write_csv(as_tibble(summary(emm)), "output/emmeans_cells.csv")
write_csv(as_tibble(summary(pairs(emm), infer = TRUE)), "output/emmeans_contrasts.csv")

# ---- DeLong tests: Meso-4 vs MesoInception-4 AUC, per method, per seed (paired frames) ----
message("DeLong AUC comparisons ...")
delong <- list()
for (mth in c("Deepfakes", "Face2Face")) {
  for (sd_ in unique(in_domain$seed)) {
    # paired on the identical test frames: rows per run-group are in dataset order,
    # so both models' frames align 1:1 (guarded by the identical-labels stopifnot)
    da <- in_domain |> filter(method == mth, seed == sd_)
    wide_a <- da |> filter(model == "meso4")
    wide_b <- da |> filter(model == "meso_inception4")
    stopifnot(identical(wide_a$label, wide_b$label))
    r1 <- roc(wide_a$label, wide_a$prob_fake, quiet = TRUE, direction = "<")
    r2 <- roc(wide_b$label, wide_b$prob_fake, quiet = TRUE, direction = "<")
    tst <- roc.test(r1, r2, method = "delong", paired = TRUE)
    delong[[paste(mth, sd_)]] <- tibble(
      method = mth, seed = as.character(sd_),
      auc_meso4 = as.numeric(auc(r1)), auc_inception = as.numeric(auc(r2)),
      delta = as.numeric(auc(r1)) - as.numeric(auc(r2)), p.value = tst$p.value
    )
  }
}
write_csv(bind_rows(delong), "output/delong.csv")

# ---- Bootstrap CIs for every reported AUC (incl. cross-dataset vs chance) -----------------
message("Bootstrap AUC confidence intervals (2000 reps each) ...")
auc_rows <- preds |>
  distinct(model, seed, train_source, eval_dataset)
cis <- list()
for (i in seq_len(nrow(auc_rows))) {
  key <- auc_rows[i, ]
  d <- preds |> semi_join(key, by = c("model", "seed", "train_source", "eval_dataset"))
  r <- roc(d$label, d$prob_fake, quiet = TRUE, direction = "<")
  ci <- ci.auc(r, method = "bootstrap", boot.n = 2000, progress = "none")
  cis[[i]] <- key |> mutate(
    n = nrow(d), auc = as.numeric(auc(r)),
    ci_lo = ci[1], ci_hi = ci[3],
    in_domain = train_source == eval_dataset,
    excludes_chance = (ci[1] > 0.5) | (ci[3] < 0.5)
  )
}
write_csv(bind_rows(cis), "output/auc_cis.csv")

message("Done — outputs in analysis/output/")
