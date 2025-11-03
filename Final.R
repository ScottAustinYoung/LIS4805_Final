# Set up of the R environment.
knitr::opts_chunk$set(echo = TRUE, message = FALSE, warning = FALSE)

# Install packages as needed, uncomment and run until I have an if statement written.
# install.packages("tidyverse")
# install.packages("caret")
# install.packages("randomForest")
# install.packages("janitor")

library(tidyverse)
library(caret)
library(randomForest)
library(janitor)

# The goal overall for this project is to create predictive models that accurately
# predict the user rating of an application released on the Google PLay Store.
# The input variables will be various attributes of the app, such as its category,
# number of installs, size, and price. The outcome variable will be the app's
# numerical user rating. This is a supervised regression task.

# Loading the dataset
raw_df <- read_csv("googleplaystore.csv")

# Cleaning column names to work easier within R.
raw_df <- raw_df %>%
  clean_names()

# Cleaning the Ratings column by removing rows which have an empty value.
clean_df <- raw_df %>%
  filter(!is.na(rating))

# Cleaning the Price column by removing the '$' symbol and converting type to numeric.
clean_df <- clean_df %>%
  mutate(price = stringr::str_remove(price, "\\$") %>% as.numeric())

# Cleaning the Size column by converting values to a uniform value between measurements of M and k.
clean_df <- clean_df %>%
  mutate(
    size_kb = case_when(
      str_detect(size, "M") ~ as.numeric(str_remove(size, "M")) * 1024,
      str_detect(size, "k") ~ as.numeric(str_remove(size, "k")),
                 TRUE ~ NA_real_)
    )
  )

# Imputing the missing Size values that were introduced in the previous step.
median_size <- median(clean_df$size_kb, na.rm = TRUE)
clean_df <- clean_df %>%
  mutate(size_kb = ifelse(is.na(size_kb), median_size, size_kb))

# Cleaning the Type column, as there is one row with a misinput value.
clean_df <- clean_df %>%
  filter(type %in% c("Free", "Paid"))

# Converting categorical columns into factors for predictive analysis models.
clean_df <- clean_df %>%
  mutate(
    type = as.factor(type),
    content_rating = as.factor(content_rating),
    category = as.factor(category)
  )

# Printing a head table of the cleaned data.
knitr::kable(head(clean_df))
