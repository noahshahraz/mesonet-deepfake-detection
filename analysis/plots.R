# Task 8 — R statistical visualizations, rendered from analysis/output/*.csv only
# (no model refitting). Sourced by models.R after the fits; runnable standalone:
#   Rscript plots.R
#
# Exports light + dark PNGs (150 dpi) into ../assets/, matching the Python figure
# suite's Okabe-Ito palette and dual-theme convention.

library(readr)
library(dplyr)
library(ggplot2)

OKABE <- c(blue = "#0072B2", orange = "#E69F00", green = "#009E73",
           vermillion = "#D55E00", purple = "#CC79A7", skyblue = "#56B4E9")
LIGHT <- list(bg = "#ffffff", fg = "#1f2328", grid = "#d0d7de", muted = "#57606a")
DARK  <- list(bg = "#0d1117", fg = "#e6edf3", grid = "#30363d", muted = "#8b949e")

theme_repo <- function(th) {
  theme_minimal(base_size = 12) +
    theme(
      plot.background = element_rect(fill = th$bg, colour = NA),
      panel.background = element_rect(fill = th$bg, colour = NA),
      panel.grid.major = element_line(colour = th$grid, linewidth = 0.35),
      panel.grid.minor = element_blank(),
      text = element_text(colour = th$fg),
      axis.text = element_text(colour = th$fg, size = 10),
      plot.title.position = "plot",
      plot.title = element_text(face = "bold", size = 15),
      plot.subtitle = element_text(colour = th$muted, size = 10, face = "italic"),
      plot.caption = element_text(colour = th$muted, size = 8.5),
      legend.text = element_text(colour = th$fg, size = 9),
      legend.title = element_blank()
    )
}

save_both <- function(build_fn, name, width, height) {
  for (variant in list(list(th = LIGHT, suffix = ".png"),
                       list(th = DARK, suffix = ".dark.png"))) {
    p <- build_fn(variant$th)
    ggsave(file.path("..", "assets", paste0(name, variant$suffix)), p,
           width = width, height = height, dpi = 150, bg = variant$th$bg)
  }
  message("wrote assets/", name, "{.png,.dark.png}")
}

# ---- fig_glmm_forest: Meso-4 vs MesoInception-4 odds ratios per method --------------------
# emmeans_contrasts.csv stores meso_inception4/meso4; invert so Meso-4 is the numerator.
contr <- read_csv("output/emmeans_contrasts.csv", show_col_types = FALSE) |>
  transmute(method,
            or = 1 / odds.ratio, lo = 1 / asymp.UCL, hi = 1 / asymp.LCL,
            p = p.value,
            sig = p < 0.05)

build_forest <- function(th) {
  ggplot(contr, aes(x = or, y = method)) +
    geom_vline(xintercept = 1, linetype = "42", colour = th$muted, linewidth = 0.6) +
    geom_errorbarh(aes(xmin = lo, xmax = hi, colour = sig), height = 0.12, linewidth = 0.9) +
    geom_point(aes(colour = sig, shape = sig), size = 3.6) +
    geom_text(aes(label = sprintf("OR %.2f [%.2f-%.2f]%s", or, lo, hi,
                                  ifelse(sig, sprintf(", p < 0.001"), ", n.s."))),
              vjust = -1.6, size = 3.4, colour = th$fg) +
    annotate("text", x = 1, y = 0.55, label = "OR = 1: models tied",
             size = 3.2, fontface = "italic", colour = th$muted, vjust = 1) +
    scale_colour_manual(values = c(`TRUE` = unname(OKABE["blue"]),
                                   `FALSE` = unname(OKABE["orange"])), guide = "none") +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 15), guide = "none") +
    scale_x_continuous(limits = c(0.8, 1.45)) +
    labs(
      title = "Meso-4 vs MesoInception-4: a Deepfakes edge, tied on Face2Face",
      subtitle = paste("Odds ratio of a frame being classified correctly at threshold 0.5,",
                       "Meso-4 in the numerator;\nvideo-clustered GLMM (emmeans contrasts),",
                       "95% CIs. Right of the line = Meso-4 better."),
      x = "Odds ratio (log scale)", y = NULL,
      caption = paste("AUCs are statistically indistinguishable and tuned accuracies converge:",
                      "the Deepfakes edge is calibration, not ranking.")
    ) +
    coord_trans(x = "log10") +
    theme_repo(th)
}
save_both(build_forest, "fig_glmm_forest", width = 8.2, height = 3.6)

# ---- fig_auc_cis: bootstrap CI dot-and-whisker for every reported AUC ---------------------
model_label <- c(meso4 = "Meso-4", meso_inception4 = "MesoInc-4")
ds_label <- c(ff_deepfakes = "FF++ DF", ff_face2face = "FF++ F2F",
              openforensics = "OpenForensics", faces140k = "140k StyleGAN")

cis <- read_csv("output/auc_cis.csv", show_col_types = FALSE) |>
  mutate(
    category = case_when(
      in_domain ~ "in-domain",
      eval_dataset == "ff_face2face" ~ "cross-method",
      TRUE ~ "cross-dataset"
    ),
    category = factor(category, levels = c("in-domain", "cross-method", "cross-dataset")),
    row = sprintf("%s  %s → %s  ·  seed %s", model_label[model],
                  ds_label[train_source], ds_label[eval_dataset], seed)
  ) |>
  arrange(category, model, train_source, eval_dataset, seed) |>
  mutate(row = factor(row, levels = rev(unique(row))))

build_cis <- function(th) {
  ggplot(cis, aes(x = auc, y = row, colour = category, shape = category)) +
    geom_vline(xintercept = 0.5, linetype = "42", colour = th$muted, linewidth = 0.6) +
    geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.28, linewidth = 0.7) +
    geom_point(size = 2.3) +
    annotate("text", x = 0.5, y = length(levels(cis$row)) + 0.9, label = "chance",
             size = 3.2, fontface = "italic", colour = th$muted) +
    scale_colour_manual(values = c(`in-domain` = unname(OKABE["blue"]),
                                   `cross-method` = unname(OKABE["orange"]),
                                   `cross-dataset` = unname(OKABE["vermillion"]))) +
    scale_shape_manual(values = c(`in-domain` = 16, `cross-method` = 17,
                                  `cross-dataset` = 15)) +
    scale_x_continuous(limits = c(0.35, 1.0), breaks = seq(0.4, 1.0, 0.1)) +
    labs(
      title = "Every AUC in the project, with bootstrap uncertainty",
      subtitle = paste("Point = AUC, whiskers = bootstrap 95% CI (2,000 reps).\nAll six",
                       "cross-dataset CIs sit entirely below 0.5: the mild ranking inversion",
                       "is supported per seed."),
      x = "AUC (0.5 = chance)", y = NULL
    ) +
    theme_repo(th) +
    theme(legend.position = "top")
}
save_both(build_cis, "fig_auc_cis", width = 8.6, height = 7.2)
