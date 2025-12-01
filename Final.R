# Set up of the R environment.
knitr::opts_chunk$set(echo = TRUE, message = FALSE, warning = FALSE)

# Install packages as needed, uncomment and run until I have an if statement written.
# install.packages("tidyverse")
# install.packages("caret")
# install.packages("randomForest")
# install.packages("janitor")
# install.packages("car")
# install.packages("ggplot2")
# install.packages("stringr")
# install.packages("xgboost")


library(tidyverse)
library(caret)
library(randomForest)
library(janitor)
library(car)
library(ggplot2)
library(stringr)
library(xgboost)

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

# Cleaning the Installs column by removing and test and converting to numeric
clean_df <- clean_df %>%
  mutate(
    installs = stringr::str_remove_all(installs, "[\\+,]") %>%
      as.numeric()
  )

# Log transformation of the reviews variable to stabalize the models.
clean_df <- clean_df %>%
  mutate(
    reviews = log(reviews + 1)
  )

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

# Added a composite variable to make price less granular in its use in the models.
clean_df <- clean_df %>%
  mutate(
    price_cat = case_when(
      price == 0 ~ "Free",
      price <= 2 ~ "Low",
      price <= 10 ~ "Medium",
      TRUE ~ "High"
    ),
    price_cat = factor(price_cat),
    review_per_install = reviews / (installs + 1)
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
    category,
    price_cat,
    review_per_install
  )

# Creating the training index, train set, and test set.
set.seed(2)
train_index <- createDataPartition(model_df$rating, p = 0.8, list = FALSE)
train_set <- model_df[train_index, ]
test_set <- model_df[-train_index, ]

# Imputing the missing Size values for both test and train data.
median_size_train <- median(train_set$size_kb, na.rm = TRUE)
train_set <- train_set %>%
  mutate(size_kb = ifelse(is.na(size_kb), median_size_train, size_kb))
median_size_test <- median(test_set$size_kb, na.rm = TRUE)
test_set <- test_set %>%
  mutate(size_kb = ifelse(is.na(size_kb), median_size_test, size_kb))

# Fixing multicollinearity with a VIF check.
vif_model <- lm(
  rating ~ reviews + size_kb + installs + price + type + content_rating + category,
  data = train_set
)
vif(vif_model)
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
  rating ~ . - type,
  data = train_set,
  method = "lm",
  trControl = train_control,
  preProcess = c("center", "scale", "nzv"),
  metric = "RMSE"
)
print(model_2)
print(model_2$results)

# Model 3 Random Forest
set.seed(2)
model_3 <- train(
  rating ~ . -type,
  data = train_set,
  method = "rf",
  trControl = train_control,
  metric = "RMSE",
  tuneGrid = expand.grid(.mtry = c(2, 4, 7)),
  importance = TRUE
)
print(model_3)
print(model_3$results)

# Model 4 on the new variables.
model_4 <- train(
  rating ~ reviews + installs + size_kb + price_cat + review_per_install + category,
  data = train_set,
  method = "lm",
  trControl = train_control,
  metric = "RMSE"
)
print(model_4)
print(model_4$results)

# Model 5 Gradient Boosting Algorithm
model_gb <- train(
  rating ~. -type,
  data = train_set,
  method = "xgbTree",
  trControl = train_control,
  metric = "RMSE",
  verbosity = 0
)

# Predictions based on the test data set
pred_1 <- predict(model_1, newdata = test_set)
metrics_1 <- postResample(pred_1, test_set$rating)

pred_2 <- predict(model_2, newdata = test_set)
metrics_2 <- postResample(pred_2, test_set$rating)

pred_3 <- predict(model_3, newdata = test_set)
metrics_3 <- postResample(pred_3, test_set$rating)

pred_4 <- predict(model_4, newdata = test_set)
metrics_4 <- postResample(pred_4, test_set$rating)

pred_gb <- predict(model_gb, newdata = test_set)
metrics_gb <- postResample(pred_gb, test_set$rating)

# Plots
plot_data1 <- data.frame(Actual = test_set$rating, Predicted = pred_1)
plot_data2 <- data.frame(Actual = test_set$rating, Predicted = pred_2)
plot_data3 <- data.frame(Actual = test_set$rating, Predicted = pred_3)
plot_data4 <- data.frame(Actual = test_set$rating, Predicted = pred_4)
plot_data5 <- data.frame(Actual = test_set$rating, Predicted = pred_gb)

# Plot 1
ggplot(plot_data1, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Model 1: Linear Regression (Basic)", 
       subtitle = paste0("RMSE: ", round(metrics_1['RMSE'], 3), " | R-Squared: ", round(metrics_1['Rsquared'], 3))) +
  theme_minimal() +
  coord_cartesian(xlim = c(1, 5), ylim = c(1, 5))

# Plot 2
ggplot(plot_data2, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Model 2: Linear Regression (All Vars)", 
       subtitle = paste0("RMSE: ", round(metrics_2['RMSE'], 3), " | R-Squared: ", round(metrics_2['Rsquared'], 3))) +
  theme_minimal() +
  coord_cartesian(xlim = c(1, 5), ylim = c(1, 5))

# Plot 3
ggplot(plot_data3, aes(x = Actual, y = Predicted)) +
  geom_point(color = "darkgreen", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Model 3: Random Forest (Best Model)", 
       subtitle = paste0("RMSE: ", round(metrics_3['RMSE'], 3), " | R-Squared: ", round(metrics_3['Rsquared'], 3))) +
  theme_minimal() +
  coord_cartesian(xlim = c(1, 5), ylim = c(1, 5))

# Plot 4
ggplot(plot_data4, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Model 4: Linear Regression (Engineered)", 
       subtitle = paste0("RMSE: ", round(metrics_4['RMSE'], 3), " | R-Squared: ", round(metrics_4['Rsquared'], 3))) +
  theme_minimal() +
  coord_cartesian(xlim = c(1, 5), ylim = c(1, 5))

# Plot 5
ggplot(plot_data5, aes(x = Actual, y = Predicted)) +
  geom_point(color = "purple", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Model 5: XGBoost (Gradient Boosting)", 
       subtitle = paste0("RMSE: ", round(metrics_gb['RMSE'], 3), " | R-Squared: ", round(metrics_gb['Rsquared'], 3))) +
  theme_minimal() +
  coord_cartesian(xlim = c(1, 5), ylim = c(1, 5))

# Plot for  variable importance
importance_data <- varImp(model_3)$importance %>% 
  as.data.frame() %>%
  rownames_to_column("Variable") %>%
  filter(!str_detect(Variable, "reviews")) %>%
  mutate(Variable = str_replace_all(Variable, "[_\\.]", " ")) %>%
  mutate(Variable = str_to_title(Variable)) %>%
  # Sort by importance
  arrange(desc(Overall)) %>%
  slice_head(n = 15)

ggplot(importance_data, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 15 Drivers of App User Rating", 
       x = "", 
       y = "Importance Score") +
  theme(
    axis.text.y = element_text(size = 10, color = "black"),
    plot.title = element_text(size = 14, face = "bold"),
    panel.grid.major.y = element_blank()
  )
