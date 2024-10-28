# Naive Bayes Classifier 

# Load libraries
library(ggplot2)
library(ROSE)
library(tidymodels)
library(discrim)
library(naivebayes)
library(embed)
tidymodels_prefer()

# Read in the data 
train <- vroom::vroom("train.csv")
test <- vroom::vroom("test.csv")

# Transform response to factor
train <- train |> 
  mutate(ACTION = as.factor(ACTION))

# Balance the data with undersampling
balanced_train <- ovun.sample(ACTION ~ ., data = train, method = "under")$data

# Take 10% of the data 
ten_perc <- balanced_train |> 
  sample_frac(0.10)

# Recipe - 0.69090
my_recipe <- recipe(ACTION ~ ., data = ten_perc) |> 
  step_mutate_at(all_numeric_predictors(), fn=factor) |> 
  step_normalize(all_numeric_predictors())

# Recipe - 0.58715
my_recipe <- recipe(ACTION ~ ., data = ten_perc) |> 
  step_mutate_at(all_numeric_predictors(), fn=factor) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_novel(all_nominal_predictors()) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_pca(all_predictors(), threshold=0.85)

# Recipe - 0.59053
my_recipe <- recipe(ACTION ~ ., data = ten_perc) |> 
  step_mutate_at(all_numeric_predictors(), fn=factor) |> 
  step_novel(all_nominal_predictors()) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_pca(all_predictors(), threshold=0.85)

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |> 
  set_mode("classification") |> 
  set_engine("naivebayes")

nb_wf <- workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(nb_model)

# Grid of values to tune over
grid_of_tuning_params <- grid_regular(Laplace(), smoothness(), levels = 5) # L^2 total tuning possibilties


# Split data for CV (5-10 groups)
folds <- vfold_cv(ten_perc, v=5, repeats = 1)

# Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds, 
            grid=grid_of_tuning_params, 
            metrics=metric_set(roc_auc))

# Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Finalize the workflow and fit it
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=ten_perc)

# Make predictions
predictions <- predict(final_wf, new_data = test, type = "prob")

# Kaggle submission 
kaggle_submission <- predictions %>%  
  bind_cols(., test) %>% 
  dplyr::select(id, .pred_1) %>% 
  rename(ACTION= .pred_1)

vroom::vroom_write(kaggle_submission, "NaiveBayesPreds.csv", delim = ",")











  
  
  
  
  