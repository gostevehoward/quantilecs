darling_1967_boundary <- function(n) 3 * sqrt(n * (log(log(n)) + 1.457) / 2) / 2

szorenyi_boundary <- function(n, alpha) {
    sqrt(n / 2 * log(pi^2 * (n-31)^2 / (3 * alpha)))
}

dkw_fixed_boundary <- function(n, alpha) sqrt(n / 2 * log(2/alpha))

bernoulli_fixed_boundary <- function(ns, alpha, g, h) {
    bernoulli_kl <- function(q, p) {
        q * log(q/p) + (1 - q) * log((1 - q) / (1 - p))
    }
    psi_star <- function(u) bernoulli_kl((g + u) / (g + h), g / (g + h))
    sapply(ns, function(n) {
        upper <- (1 - 1e-10) * n * h
        if (n * psi_star(upper / n) < log(1/alpha)) {
            n * h
        } else {
            uniroot(function(x) n * psi_star(x/n) - log(1/alpha),
                    c(0, upper))$root
        }
    })
}

darling_1968_bound <- function(ns) {
    sqrt((ns + 1) * (2 * log(ns) + .601))
}

LINE_TYPES <- c(
    darling_1967='dashed',
    beta_binomial='dashed',
    subgamma_stitching='dashed',
    szorenyi='solid',
    dkw='dashed',
    bernoulli_fixed='dotted',
    double='solid',
    darling_1968='solid',
    empirical_lil='solid'
)

LABELS <- c(
    darling_1967='Darling & Robbins (1967)',
    beta_binomial='Beta\u00adBinomial, our Theorem 1(b)',
    subgamma_stitching='Stitching, our Theorem 1(a)',
    szorenyi='Szorenyi et al. (2015)',
    dkw='Pointwise DKW (Massart, 1990)',
    bernoulli_fixed='Pointwise Bernoulli (Hoeffding, 1963)',
    double='Our Theorem 3',
    darling_1968='Darling & Robbins (1968)',
    empirical_lil='Uniform DKW, our Corollary 2'
)

make_quantile_plot_data <- function(p, max_n) {
    n <- unique(round(exp(seq(log(32), log(max_n), length.out=500))))
    g <- p
    h <- 1 - p
    bound_data <- data.frame(
        n=n,
        darling_1967=darling_1967_boundary(n),
        beta_binomial=beta_binomial_mixture_bound(n * g * h, .05, 32 * g * h,
                                                  g, h, is_one_sided=FALSE),
        subgamma_stitching=poly_stitching_bound(n * g * h, .025, 32 * g * h,
                                                c=(h - g) / 3),
        szorenyi=szorenyi_boundary(n, .05),
        dkw=dkw_fixed_boundary(n, .05),
        bernoulli_fixed=bernoulli_fixed_boundary(n, .025, g, h),
        double=double_stitching_bound(p, n, .05, 32),
        darling_1968=darling_1968_bound(n),
        empirical_lil=empirical_process_lil_bound(n, .05, 32) * n
    )
    label_order <- names(sort(tail(bound_data, 1)[-1], decreasing=TRUE))
    plot_data <- (
        bound_data
        %>% gather('type', 'bound', -n)
        %>% mutate(bound=ifelse(p + bound/n >= 1, NA, bound),
                   type=factor(as.character(type), levels=label_order),
                   p=p)
    )
    plot_data
}

plot_quantile_cs <- function(ps=c(.05, .5, .95), max_n=1e6, save=F) {
    colors <- rep(COLORS, length.out=length(LINE_TYPES))
    names(colors) <- names(LINE_TYPES)

    plot_dfs <- lapply(ps, function(p) make_quantile_plot_data(p, max_n))
    plot_data <- do.call(rbind, plot_dfs)
    middle_index <- ceiling(length(ps) / 2)
    label_order <- levels(plot_dfs[[middle_index]]$type)
    plot_data <- mutate(plot_data, type=factor(as.character(type),
                                               levels=label_order,
                                               labels=LABELS[label_order]))
    plot <- (
        ggplot(plot_data, aes(n, bound / sqrt(n), color=type, linetype=type))
        + geom_line(size=0.75)
        + facet_grid(~ p, scale='free_y',
                     labeller=label_bquote(cols=italic(p) == .(p)))
        + scale_x_log10(bquote(italic(t)),
                        breaks=10^(2:round(log10(max_n))),
                        labels=scales::trans_format("log10",
                                                    scales::math_format(10^.x)))
        + scale_y_continuous(name=bquote(italic(u[t] * sqrt(t))))
        + scale_color_manual(NULL, values=unname(colors[label_order]))
        + scale_linetype_manual(NULL, values=unname(LINE_TYPES[label_order]))
        + coord_cartesian(xlim=c(32, max_n), ylim=c(0, 5), expand=FALSE)
        + guides(color=guide_legend(ncol=2))
        + report_theme
        + theme(legend.position='top', legend.key.width=unit(2, 'line'))
        + margins_pt(right=7)
    )
    if (save) {
        save_plot(
            plot,
            'quantile_bounds',
            'paper',
            list(paper=c(PAPER_WIDTH, 4)),
            tag=max_n
        )
    }
    plot
}

plot_tuning <- function(p=.5, max_n=1e5, save=F) {
    n <- unique(round(exp(seq(log(32), log(max_n), length.out=500))))
    plot_dfs <- lapply(
        c(100, 1000, 10000),
        function(m) {
            data.frame(n=n,
                       label=sprintf('italic(m) == "%s    "',
                                     prettyNum(m, big.mark=',')),
                       bound=beta_binomial_mixture_bound(
                           n * p * (1 - p), .05, m * p * (1 - p), p, 1 - p))
        }
    )
    plot_data <- (
        do.call(rbind, plot_dfs)
        %>% mutate(bound=ifelse(p + bound/n >= 1, NA, bound))
    )
    plot <- (
        ggplot(plot_data, aes(n, bound / sqrt(n), color=label))
        + geom_line(size=0.75)
        + scale_x_log10(bquote(italic(t)),
                        breaks=10^(2:round(log10(max_n))),
                        labels=scales::trans_format("log10",
                                                    scales::math_format(10^.x)))
        + scale_y_continuous(name=bquote(italic(u[t] * sqrt(t))))
        + scale_color_manual(NULL, values=unname(COLORS),
                             labels=scales::parse_format())
        + coord_cartesian(xlim=c(32, max_n), ylim=c(0, 3), expand=FALSE)
        + report_theme
        + theme( legend.key.width=unit(2, 'line'))
        + margins_pt(right=7)
    )
    if (save) {
        save_plot(
            plot,
            'tuning',
            'paper',
            list(paper=c(PAPER_WIDTH, 2))
        )
    }
    plot
}

make_intro_plot <- function(max_t=1e4, alpha=0.05, p=0.9,
                            cdf_times=c(1e2, 1e3, 1e4), save=FALSE) {
    set.seed(83882985)
    t <- 1:max_t
    Xt <- rcauchy(t)
    upper_radii <- beta_binomial_mixture_bound(
        p * (1 - p) * t, alpha, 100 * p * (1 - p), p, 1 - p) / t
    lower_radii <- beta_binomial_mixture_bound(
        p * (1 - p) * t, alpha, 100 * p * (1 - p), 1 - p, p) / t
    quantiles <- sapply(t, function(i) {
        lower_p <- p - lower_radii[i]
        upper_p <- p + upper_radii[i]
        samples <- sort(Xt[1:i])
        c(ifelse(lower_p > 0, samples[floor(lower_p * i) + 1], NA),
          samples[round(p * i)],
          ifelse(upper_p < 1, samples[ceiling(upper_p * i)], NA))
    })
    lower_bounds <- quantiles[1,]
    point_estimates <- quantiles[2,]
    upper_bounds <- quantiles[3,]
    print(c(max(lower_bounds, na.rm=TRUE), min(upper_bounds, na.rm=TRUE)))

    plot_data <- (
        data.frame(t=t,
                   lower=lower_bounds,
                   upper=upper_bounds,
                   point=point_estimates)
        %>% gather(key, value, -t))
    final_diameter <- tail(upper_bounds, 1) - tail(lower_bounds, 1)
    fixed_plot <- (
        ggplot(plot_data, aes(t, value, linetype=key))
        + geom_hline(yintercept=qcauchy(p), color='grey')
        + geom_line()
        + scale_linetype_manual(values=c('solid', 'dotted', 'solid'))
        + scale_x_continuous(bquote(paste('Number of samples, ', italic(t))))
        + scale_y_continuous('Confidence bounds for 90%ile')
        + guides(linetype=FALSE)
        + coord_cartesian(
              xlim=c(1, max_t),
              ylim=c(tail(lower_bounds, 1) - 4 * final_diameter,
                     tail(upper_bounds, 1) + 4 * final_diameter),
              expand=FALSE)
        + report_theme
        + theme(panel.grid.major=element_blank(),
                panel.grid.minor=element_blank())
        + margins_pt(right=7)
    )

    cdf_dfs <- lapply(cdf_times, function(time) {
        samples <- Xt[1:time]
        radius <- empirical_process_lil_bound(time, alpha, min(cdf_times))
        plot_x <- seq(-7, 7, length.out=1000)
        cdf <- ecdf(samples)(plot_x)
        (
            data.frame(time=time,
                       x=plot_x,
                       lower=pmax(0, cdf - radius),
                       point=cdf,
                       truth=pcauchy(plot_x),
                       upper=pmin(1, cdf + radius))
            %>% gather(key, value, -x, -time))
    })
    plot_data <- do.call(rbind, cdf_dfs)

    label_fn <- label_bquote(italic(t) == .(prettyNum(time, big.mark=',')))
    cdf_plot <- (
        ggplot(plot_data, aes(x, value, linetype=key, color=key))
        + geom_line()
        + facet_grid(time ~ ., labeller=label_fn)
        + scale_x_continuous(bquote(italic(x)))
        + scale_y_continuous('CDF confidence band', breaks=c(0, .5, 1))
        + scale_linetype_manual(values=c('solid', 'dotted', 'solid', 'solid'))
        + scale_color_manual(values=c('black', 'black', 'grey', 'black'))
        + guides(linetype=FALSE, color=FALSE)
        + coord_cartesian(xlim=c(-7, 7), ylim=c(0, 1), expand=FALSE)
        + report_theme
        + theme(panel.grid.major=element_blank(),
                panel.grid.minor=element_blank(),
                panel.spacing=unit(1, "lines"))
    )

    final_plot <- arrangeGrob(fixed_plot, cdf_plot, ncol=2)

    if (save) {
        save_plot(
            final_plot,
            'intro',
            'paper',
            list(paper=c(PAPER_WIDTH, 3))
        )
    }
    grid.arrange(final_plot)
}
