qbai_cpp = Module("quantile_bai_cpp")

run_simulation <- function(num_replications, max_rounds, quantile_p, arm_type,
                           num_threads=1, epsilon=0.025, effect=NULL,
                           num_arms=10) {
    config <- qbai_cpp$SimulationConfig$new(num_replications, max_rounds,
                                            num_threads, quantile_p, epsilon)
    if (arm_type == 'bernoulli') {
        config$bernoulli_model(num_arms, 0.05)
    } else if (arm_type == 'uniform') {
        if (is.null(effect)) {
            effect <- 2 * epsilon
        }
        config$uniform_model(num_arms, effect)
    } else if (arm_type == 'cauchy') {
        if (is.null(effect)) {
            epsilon_quantile_dist <- (
                qcauchy(quantile_p + epsilon) - qcauchy(quantile_p))
            effect <- 2 * epsilon_quantile_dist
        }
        config$cauchy_model(num_arms, effect)
    } else if (arm_type == 'normal_spread') {
        config$normal_spread_model(num_arms);
    } else error(paste('Unknown arm type', arm_type))

    if (num_arms == 2) {
        config$enable_ab_test()
    }

    results <- qbai_cpp$run_simulation(config)
    result_data_rows <- lapply(1:length(results), function(replication) {
        rep_results <- results[[replication]]
        lapply(names(rep_results), function(strategy_name) {
            data.frame(replication=replication,
                       strategy=strategy_name,
                       num_rounds=rep_results[[strategy_name]]$num_rounds,
                       chosen_arm=rep_results[[strategy_name]]$chosen_arm,
                       stringsAsFactors=FALSE)
        })
    })
    result_data <- (
        flatten_dfr(result_data_rows)
        %>% mutate(strategy=factor(strategy)))
    result_data
}

run_paper_simulations <- function(num_replications=64, max_rounds=1e7,
                                  num_threads=4, num_arms=10, epsilon=0.025,
                                  effect=NULL, save=FALSE) {
    if (save) {
        filename <- sprintf('build/simulations_%d_%d.csv',
                            num_arms, num_replications)
        if (!dir.exists(dirname(filename))) {
            cat(sprintf('Directory %s does not exist\n', basename(filename)))
            stop()
        }
    }
    param_data <- expand.grid(
        quantile_p=c(.05, seq(.1, .9, .1), .95),
        arm_type=c('uniform', 'cauchy', 'normal_spread'))
    if (num_arms == 2) {
        param_data <- filter(
            param_data,
            !(arm_type == 'normal_spread' & quantile_p == 0.5))
    }
    results <- lapply(1:nrow(param_data), function(index) {
        row <- param_data[index,]
        print(row)
        result_df <- run_simulation(num_replications, max_rounds,
                                    quantile_p=row$quantile_p,
                                    arm_type=row$arm_type,
                                    num_threads=num_threads,
                                    epsilon=epsilon, effect=effect,
                                    num_arms=num_arms)
        rownames(row) <- NULL
        cbind(row, result_df)
    })
    result_data <- do.call(rbind, results)

    if (save) {
        cat(sprintf('Saving results to %s...\n', filename))
        write.csv(result_data, file=filename)
    }

    result_data
}

run_ab_simulations <- function(num_replications=100, max_rounds=1e5,
                               num_threads=4, epsilon=0.025, save=FALSE) {
    run_paper_simulations(num_replications=num_replications,
                          max_rounds=max_rounds, num_threads=num_threads,
                          num_arms=2, epsilon=epsilon, save=save)
}

check_for_unfinished_runs <- function(result_data) {
    if (any(result_data$chosen_arm == -1)) {
        warning('Some runs did not finish')
    }
}

print_diagnostics <- function(result_data) {
    print(result_data %>% filter(arm_type == 'normal_spread')
          %>% group_by(strategy, quantile_p)
          %>% summarize(mean_arm_0=mean(chosen_arm == 0))
          %>% spread(strategy, mean_arm_0)
          %>% as.data.frame)
    print(result_data
          %>% group_by(arm_type, quantile_p, strategy)
          %>% summarize(mean_samples=mean(num_rounds))
          %>% spread(strategy, mean_samples)
          %>% as.data.frame)
    print(result_data
          %>% group_by(arm_type, quantile_p, strategy)
          %>% summarize(mean_samples=mean(num_rounds))
          %>% spread(strategy, mean_samples)
          %>% mutate_at(vars(-quantile_p, -arm_type, -`LUCB BB`),
                        funs(. / `LUCB BB`))
          %>% select(-`LUCB BB`)
          %>% as.data.frame)
}

make_simulation_plots <- function(csv_filename, save=FALSE) {
    colors = COLORS[c(2,1,3,4,5,6)]
    result_data <- read.csv(csv_filename)
    check_for_unfinished_runs(result_data)
    print(table(result_data$chosen_arm))
    print_diagnostics(result_data)

    mean_data <- (
        result_data
        %>% group_by(quantile_p, arm_type, strategy)
        %>% summarize(mean_rounds=mean(num_rounds),
                      se_of_mean=sd(num_rounds) / sqrt(n()),
                      num_complete=sum(chosen_arm > -1),
                      num_wrong=sum(chosen_arm > 0))
        %>% ungroup()
        %>% mutate(arm_type=factor(
                       as.character(arm_type),
                       levels=c('uniform', 'cauchy', 'normal_spread'),
                       labels=c('Uniform', 'Cauchy', 'Normal spread')))
        %>% mutate(strategy=factor(
                       as.character(strategy),
                       levels=c('Doubled Max-Q',
                                'QPAC', 'QPAC BB',
                                'LUCB DKW', 'LUCB Stitched', 'LUCB BB'),
                       labels=c('Doubled Max\u00adQ    ',
                                'QPAC DKW   ', 'QPAC B\u00adB (ours)    ',
                                'QLUCB DKW (ours)    ',
                                'QLUCB Stitched (ours)',
                                'QLUCB B\u00adB (ours)')))
    )
    print(mean_data)
    print(mean_data %>% group_by(arm_type, strategy)
          %>% summarize(sum(num_complete), sum(num_wrong)))
    plot <- (
        ggplot(mean_data, aes(quantile_p, mean_rounds, color=strategy,
                              shape=strategy))
        + geom_line(size=0.75, linetype='dashed')
        + geom_point(size=1.5)
        + facet_wrap(~ arm_type, ncol=3)
        + scale_x_continuous('Quantile', breaks=c(0, .25, .5, .75, 1),
                             labels=c('0', '0.25', '0.5', '0.75', '1'))
        + scale_y_log10('Mean sample size',
                        breaks=10^(3:6),
                        labels=scales::trans_format("log10",
                                                    scales::math_format(10^.x)))
        + scale_color_manual(NULL, values=colors)
        + scale_shape_manual(NULL, values=0:5)
        + coord_cartesian(xlim=c(0, 1), ylim=c(2e3, 6e6), expand=FALSE)
        + guides(color=guide_legend(ncol=3))
        + report_theme
        + theme(legend.position='top', legend.key.width=unit(1.6, 'line'))
    )
    if (save) {
        save_plot(plot, 'bai_simulations', 'paper', list(paper=c(6.5, 3)))
    }
    plot
}

make_ab_plots <- function(csv_filename, save=FALSE) {
    result_data <- read.csv(csv_filename)
    check_for_unfinished_runs(result_data)
    print_diagnostics(result_data)

    ratio_data <- (
        result_data
        %>% select(replication, quantile_p, arm_type, strategy, num_rounds)
        %>% spread(strategy, num_rounds)
        %>% group_by(quantile_p, arm_type)
        %>% summarize(mean_ratio=mean(ABTest / `LUCB BB`),
                      se_of_mean=sd(ABTest/ `LUCB BB`) / sqrt(n()))
        %>% mutate(arm_type=factor(
                       as.character(arm_type),
                       levels=c('uniform', 'cauchy', 'normal_spread'),
                       labels=c('Uniform', 'Cauchy', 'Normal spread')))
        %>% ungroup()
    )
    print(ratio_data)
    print(sprintf('Largest SE: %.4f\n', max(ratio_data$se_of_mean)))
    plot <- (
        ggplot(ratio_data, aes(quantile_p, mean_ratio))
        + geom_hline(yintercept=1, linetype='dashed')
        + geom_line(size=0.75)
        + geom_point(size=1.5)
        + facet_wrap(~ arm_type, ncol=3)
        + scale_x_continuous('Quantile', breaks=c(0, .25, .5, .75, 1),
                             labels=c('0', '0.25', '0.5', '0.75', '1'))
        + scale_y_continuous('Relative sample size',
                             breaks=seq(0, 1, 1/4))
        + coord_cartesian(xlim=c(0, 1), ylim=c(0, 1.2), expand=FALSE)
        + report_theme
    )
    if (save) {
        save_plot(plot, 'ab_simulations', 'paper', list(paper=c(6.5, 2)))
    }
    plot
}
