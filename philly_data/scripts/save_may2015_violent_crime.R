# Create npz file
library(reticulate)
load("data/crime_counts.RData")
load("data/tract_adjacency.RData")
tmp <- as.data.frame(viol_crimes)

x <- tmp[,"2015_05"]
names(x) <- tmp[,"TRACT"]
x <- x[rownames(W_tract)]

py_main <- import_main()
py_main$w_tract <- as.array(W_tract)
py_main$counts_tract <- as.array(as.integer(x))
py_main$tract_filename <- "crime_tracts.npz"
py_run_string(
"
import numpy as np
np.savez_compressed(tract_filename, w = w_tract, counts = counts_tract)
"
)
