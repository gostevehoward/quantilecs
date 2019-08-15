COLORS <- brewer.pal(8, 'Dark2')
PAPER_WIDTH <- 6.5
GG_FAMILY <- 'CM Roman'

report_theme <- theme_bw(base_family=GG_FAMILY) + theme(
  panel.grid.minor=element_blank(),
  axis.title=element_text(size=10),
  axis.text=element_text(size=10),
  legend.title=element_text(size=10, face='plain'),
  legend.text=element_text(size=10),
  plot.title=element_text(size=10, hjust=0.5),
  panel.background=element_rect(fill='transparent', color=NA),
  plot.background=element_rect(fill='transparent', color=NA),
  legend.background=element_rect(fill='transparent', color=NA),
  legend.box.background=element_rect(fill='transparent', color=NA),
  legend.key=element_rect(fill='transparent', color=NA)
)


save_plot <- function(plot, base_name, format, dimensions, tag=NULL,
                      filename=NULL) {
    stopifnot(format %in% names(dimensions))
    dims <- dimensions[[format]]
    if (!is.null(tag)) {
        base_name <- sprintf('%s_%s', base_name, tag)
    }
    if (is.null(filename)) {
        filename <- sprintf('build/%s_%s.pdf', base_name, format)
    }
    cat(sprintf('Saving to %s...\n', filename))
    dir.create('build', showWarnings=FALSE)
    dir.create(dirname(filename), showWarnings=FALSE)
    ggsave(filename, plot, width=dims[1], height=dims[2])
    embed_fonts(filename)
}

margins_pt <- function(top=5.5, right=5.5, bottom=5.5, left=5.5) {
    theme(plot.margin=unit(c(top, right, bottom, left), 'pt'))
}
