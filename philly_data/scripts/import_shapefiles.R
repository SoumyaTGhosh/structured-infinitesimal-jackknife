# script to import shape files in to R
# Shape files were downloaded from opendataphilly.org

library(raster)


blocks <- shapefile("raw_data/shapefiles/Census_Blocks_2010-shp/a0937081-eccb-4da9-a95f-10395c086c932020329-1-lcydjk.c7huf.shp")
block_groups <- shapefile("raw_data/shapefiles/Census_Block_Groups_2010-shp/Census_Block_Groups_2010.shp")
tracts <- shapefile("raw_data/shapefiles/Census_Tracts_2010-shp/c16590ca-5adf-4332-aaec-9323b2fa7e7d2020328-1-1jurugw.pr6w.shp")
psa <- shapefile("raw_data/shapefiles/Boundaries_PSA-shp/1d53400f-66f7-45b3-86fe-6aca8d7c3b5b2020329-1-lutzgd.vpipe.shp")
police_div <- shapefile("raw_data/shapefiles/Boundaries_Division-shp/f47f905d-0cec-4f66-9b89-1283516fe4a32020328-1-19q4rk0.59l9.shp")




save(blocks, file = "data/spatial_polygons/blocks.RData")
save(block_groups,file = "data/spatial_polygons/block_groups.RData")
save(tracts, file = "data/spatial_polygons/tracts.RData")
save(psa, file = "data/spatial_polygons/psa.RData")
save(police_div, file = "data/spatial_polygons/police_div.RData")
