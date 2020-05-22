This repository contains an R package with all code necessary to reproduce
simulation results and all plots from

Howard, S. R. and Ramdas, A. (2019), [Sequential estimation of quantiles with
applications to A/B-testing and best-arm
identification](https://arxiv.org/abs/1906.09712), preprint, arXiv:1906.09712.

You can install the package like so:

```R
install.packages('devtools')
devtools::install_github('gostevehoward/quantilecs')
```

Building the simulation code requires a C++ compiler with C++14 support.

All output files are written to the `build/` subdirectory of the working
directory. Simulations take a while to run; my results CSVs are under
`inst/extdata`. You can grab the installed paths to these files with

```R
# For Figure 5:
system.file('extdata/simulations_10_64.csv', package='quantilecs')
# For Figure 6:
system.file('extdata/simulations_2_256.csv', package='quantilecs')
```

* `make_intro_plot(save=TRUE)` reproduces Figure 1.
* `plot_quantile_cs(save=TRUE)` reproduces Figure 2.
* `plot_tuning(save=TRUE)` reproduces Figure 3.
* `run_paper_simulations(save=TRUE)` runs the simulations for Figure 5, writing
  results to `build/simulations_10_64.csv`. This takes a couple of hours on my
  four-core MacBook Pro.
* `run_ab_simulations(save=TRUE)` runs the simulations for Figure 6, writing
  results to `build/simulations_2_256.csv`. This also takes some time.
* `make_simulation_plots('path/to/csv', save=TRUE)` reproduces Figure 5.
* `make_ab_plots('path/to/csv', save=TRUE)` reproduces Figure 6.

To run the C++ unit tests, run `make -C test runtests` at the command line. You
will need to have the `BH` and `confseq` R packages installed, since we link
against C++ headers from those libraries.
