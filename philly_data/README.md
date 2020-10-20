## Preparing Philadelphia crime dataset

In order to replicate the analysis of the Philadelphia crime data, you need to download several files from opendataphilly.org
The necessary files and organization are described below.

Then run the following R scripts in this order:
1. scripts/import_shapefiles.R : this imports the shapefiles downloaded from opendataphilly.org and saves them as .RData objects in the data/spatial_polygons sub-directory
2. scripts/aggregate_crimes.R : this creates a tbl containing the monthly counts of violent and non-violent crimes in every census tract from 2006 to 2018
3. scripts/save_may2015_violent_crime.R : this pulls out the crime counts from May 2015 (used in the paper) and saves it as a .npz file, which is written to this directory
