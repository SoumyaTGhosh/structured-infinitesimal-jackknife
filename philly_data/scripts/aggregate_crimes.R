library(tidyverse)
library(lubridate)
library(sp)
library(spdep)
library(raster)


load("data/spatial_polygons/tracts.RData")

# get the adjacency matrix for the tracts
tract_names <- paste0("tr.", tracts@data$NAME10)

tract_poly <- poly2nb(tracts)
n_tract <- nrow(tracts)
tmp_w <- matrix(data = 0, nrow = n_tract, ncol = n_tract)
for(i in 1:n_tract){
  tmp_w[i, tract_poly[[i]]] <- 1
}
W_tract<- tmp_w + t(tmp_w) - tmp_w * t(tmp_w)
rownames(W_tract) <- tract_names
colnames(W_tract) <- tract_names

tract_data <- 
  as.tbl(tracts@data) %>%
  mutate("TRACT" = paste0("tr.", NAME10))

for(year in 2006:2018){
  assign(paste0("raw_crime_", year), read_csv(file = paste0("raw_data/crime_incidents/", year, "_incidents_part1_part2.csv")))
}

raw_crime_all <- bind_rows(raw_crime_2006, raw_crime_2007, raw_crime_2008, raw_crime_2009,
                           raw_crime_2010, raw_crime_2011, raw_crime_2012, raw_crime_2013,
                           raw_crime_2014, raw_crime_2015, raw_crime_2016, raw_crime_2017, raw_crime_2018)

# A function to figure out in which tract each crime incident occured
get_tract <- function(tmp_x, tmp_y){
  tmp_poly <- tracts
  tmp_poly_names <- paste0("tr.", tracts@data$NAME10)
  pts <- SpatialPoints(cbind(tmp_x, tmp_y))
  crs(pts) <- crs(tmp_poly)
  
  tmp_list <- sp::over(tmp_poly, pts, returnList = TRUE)
  output <- rep(NA_character_, times = length(tmp_x))
  for(i in 1:length(tmp_list)){
    if(length(tmp_list[[i]]) > 0){
      output[tmp_list[[i]]] <- tmp_poly_names[i]
    }
  }
  return(output)
}


viol_nonviol_crime <-
  raw_crime_all %>%
  mutate(viol_nonviol = case_when(ucr_general <= 400 | ucr_general == 800 | ucr_general == 1700 ~ "viol",
                                  ucr_general == 600 | ucr_general == 500 ~ "non_viol"),
         MONTH = month(dispatch_date),
         MONTH = case_when(MONTH == 1 ~ "01",
                           MONTH == 2 ~ "02",
                           MONTH == 3 ~ "03",
                           MONTH == 4 ~ "04",
                           MONTH == 5 ~ "05",
                           MONTH == 6 ~ "06",
                           MONTH == 7 ~ "07",
                           MONTH == 8 ~ "08",
                           MONTH == 9 ~ "09",
                           MONTH == 10 ~ "10",
                           MONTH == 11 ~ "11",
                           MONTH == 12 ~ "12"),
         YEAR = year(dispatch_date),
         YEAR_MONTH = paste0(YEAR, "_", MONTH)) %>%
  filter(!is.na(viol_nonviol)) %>%
  mutate(TRACT = get_tract(point_x, point_y)) %>%
  dplyr::select(MONTH, YEAR, YEAR_MONTH, TRACT, viol_nonviol, ucr_general)


year_list <- 2006:2018
month_list <- c("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12")
year_month <- sort(as.vector(outer(year_list, month_list, FUN = paste, sep = "_")))

viol_crimes <-
  viol_nonviol_crime %>%
  filter(viol_nonviol == "viol") %>%
  group_by(TRACT, YEAR_MONTH) %>%
  summarize(count = n()) %>%
  ungroup() %>%
  spread(key = YEAR_MONTH, value = count, fill = 0) %>%
  right_join(y = tract_data, by = "TRACT")
nonviol_crimes <-
  viol_nonviol_crime %>%
  filter(viol_nonviol == "non_viol") %>%
  group_by(TRACT, YEAR_MONTH) %>%
  summarize(count = n()) %>%
  ungroup() %>%
  spread(key = YEAR_MONTH, value = count, fill = 0) %>%
  right_join(y = tract_data, by = "TRACT")


tract_info_names <- c("TRACT", "ALAND10", "INTPTLAT10", "INTPTLON10")

viol_names <- c(tract_info_names, sort(intersect(names(viol_crimes), year_month)))
nonviol_names <- c(tract_info_names, sort(intersect(names(nonviol_crimes), year_month)))

viol_crimes <-
  viol_crimes %>%
  dplyr::select(all_of(viol_names))
nonviol_crimes <-
  nonviol_crimes %>%
  dplyr::select(all_of(nonviol_names))

save(viol_crimes, nonviol_crimes, file = "data/crime_counts.RData")
save(W_tract, file = "data/tract_adjacency.RData")

write.csv(W_tract, file = "data/tract_adjacency.csv")
