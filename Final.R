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
                 TRUE ~ NA_real_
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

# Selection of the variables used in our models.
model_df <- clean_df %>%
  select(
    rating,
    reviews,
    size_kb,
    installs,
    type,
    price,
    content_rating,
    category
  ) %>%
  na.omit()

# Creating the training index, train set, and test set.
set.seed(2)
train_index <- createDataPartition(model_df$rating, p = 0.6, list = FALSE)
train_set <- model_df[train_index, ]
test_set <- model_df[-train_index, ]

# Ten-fold cross-validation setup.
train_control <- trainControl(method = "cv", number = 10)

# Fitted Model Section.
# Model 1 Linear Regression between ratings and reviews.
set.seed(2)
model_1 <- train(
  rating ~ reviews + installs,
  data = train_set,
  method = "lm",
  trControl = train_control,
  metric = "RMSE"
)
print(model_1)
print(model_1$results)

# Model 2 Linear Regression across all variables.
set.seed(2)
model_2 <- train(
  rating ~ .,
  data = train_set,
  method = "lm",
  trControl = train_control,
  metric = "RMSE"
)
print(model_2)
print(model_2$results)

# Model 3 Random Forest
set.seed(2)
model_3 <- train(
  rating ~ .,
  data = train_set,
  method = "rf",
  trControl = train_control,
  metric = "RMSE",
  tuneGrid = expand.grid(.mtry = c(2, 4, 7)),
  importance = TRUE
)
print(model_3)
print(model_3$results)
