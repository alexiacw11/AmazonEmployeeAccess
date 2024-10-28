# Random Forest

# Load libraries
library(ggplot2)
library(ROSE)
library(tidymodels)

# Read in the data 
train <- vroom::vroom("train.csv")
test <- vroom::vroom("test.csv")

# Check for any nas in [train/test].csv
any(is.na(train))
any(is.na(test))


# Take a look at the data
glimpse(train)


# Transform response to factor
train <- train |> 
  mutate(ACTION = as.factor(ACTION))

# Look at distribution of response variable
train |> 
  dplyr::select(ACTION) |> 
  summarize(count = n())

# Plot of response variable - highly imbalanced.
ggplot(data = train, mapping = aes(x=ACTION)) + geom_bar(fill = "forestgreen") +
  xlab("Action") + ylab("Count")

# Balance the data with undersampling
balanced_train <- ovun.sample(ACTION ~ ., data = train, method = "under")$data

# Balanced amount of observations which may be helpful for regression tree
ggplot(data = as.data.frame(balanced_train), mapping = aes(x=ACTION)) + geom_bar(fill = "forestgreen") +
  xlab("Action") + ylab("Count")

# Create recipe
my_recipe <- recipe(ACTION ~ ., data = balanced_train) |> 
  step_mutate_at(all_numeric_predictors(), fn=factor)
  
prepped <- prep(my_recipe) |> 
  juice()

prepped


# Create model
my_model <- rand_forest(mtry=tune(), min_n = tune(), trees = 1000) |> 
  set_engine("ranger") |> 
  set_mode("classification")

# Create workflow with model and recipe


# Set up grid of tuning values

# Set up K-fold CV

# Find best tuning parameters

# Finalize workflow and predict






