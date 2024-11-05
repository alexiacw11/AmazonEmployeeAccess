# Random Forest

# Load libraries
library(ggplot2)
library(ROSE)
library(tidymodels)
library(embed)

# Parallel computation
# library(doParallel)
# num_cores <- parallel::detectCores() #How many cores do I have? 
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)

# Read in the data 
train <- vroom::vroom("train.csv")
test <- vroom::vroom("test.csv")

# Check for any nas in [train/test].csv
any(is.na(train))
any(is.na(test))

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

# Take 10% of the data 
ten_perc <- balanced_train |> 
  sample_frac(0.10)

# Create recipe
my_recipe <- recipe(ACTION ~ ., data = ten_perc) |> 
  step_mutate_at(all_numeric_predictors(), fn=factor) |> 
  step_normalize(all_numeric_predictors())

# Another recipe - 0.64858
# 0.88458 with full data
# Lets try with ten_perc balanced now, decreased  0.64858
# Full balanced decreased too 0.84830
my_recipe <- recipe(ACTION ~ ., data = balanced_train) |> 
  step_mutate_at(all_numeric_predictors(), fn=factor) |>
  step_zv(all_numeric_predictors()) |>
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |> 
  step_nzv(all_numeric_predictors()) |> 
  step_normalize(all_numeric_predictors()) 

# Another recipe - 0.64900 with 10 perc balanced data
# 0.88495 with full data
my_recipe <- recipe(ACTION ~ ., data = train) |> 
  step_mutate_at(all_numeric_predictors(), fn=factor) |>
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) 

# Create model
rf_model <- rand_forest(mtry=tune(), min_n = tune(), trees = 975) |> 
  set_engine("ranger") |> 
  set_mode("classification")

# Create workflow with model and recipe
rf_wf <- workflow() |> 
  add_model(rf_model) |> 
  add_recipe(my_recipe)
  
# Grid of tuning params
grid_of_tuning_params <- grid_regular(mtry(range=c(1,10)), min_n()) # L^2 total tuning possibilties

# Split data for CV
folds <- vfold_cv(train, v=5, repeats = 1)

# Split data for CV if using other
folds <- vfold_cv(train, v=5, repeats = 1)

# Run the CV
CV_results <- rf_wf %>%
  tune_grid(resamples=folds, 
            grid=grid_of_tuning_params, 
            metrics=metric_set(roc_auc))

# Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Finalize the workflow and fit it
final_wf <- rf_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=train)

# Make predictions
predictions <- predict(final_wf, new_data = test, type = "prob")
any(is.na(predictions))

# Kaggle submission 
kaggle_submission <- predictions %>%  
  bind_cols(., test) %>% 
  dplyr::select(id, .pred_1) %>% 
  rename(ACTION= .pred_1)

vroom::vroom_write(kaggle_submission, "RandomForestPreds.csv", delim = ",")

# Stop parallel computting
# stopCluster(cl)
# showConnections(all = TRUE)
# closeAllConnections()
# gc()  # This will clean up memory from any closed connections
# plan(sequential)




