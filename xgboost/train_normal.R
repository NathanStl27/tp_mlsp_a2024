#### Script pour entrainer le XGBoost sur les données de revues et de produits Amazon
options(scipen = 999) # empêcher notation scientifique
library(tidyverse)
library(xgboost)
library(scales)

### Préparation des données ----------------------------------------------------

## Importer les données
data_0_5000_raw <- read.csv("data/CCO_5000inference_train.csv")
data_5000_10000_raw <- read.csv("data/NSL_5000_10000_inference_train.csv")
data_10000_15000_raw <- read.csv("data/NSL_10000_15000_inference_train.csv")
data_15000_20000_raw <- read.csv("data/NSL_15000_20000_inference_train.csv")
data_full_raw <- bind_rows(
  data_0_5000_raw, data_5000_10000_raw, 
  data_10000_15000_raw, data_15000_20000_raw
  )


## Filtrer les valeurs extremes/aberrantes
# Couper rating_number à 25K
# Couper helpful_vote à 10
# Couper le prix à 500$
data_full <- data_full_raw %>% 
  mutate(
    rating = rating-1, # ajustement pour algo xgboost
    predicted_llm_rating = predicted_llm_rating-1,
    verified_purchase = ifelse(verified_purchase, 1, 0)
  )

## Normaliser les variables
data_normal_scaled <- data_full %>% 
  mutate(across(c(helpful_vote, average_rating, rating_number, price), ~rescale(.)))

## Train/Valid split
set.seed(12345)
val_ind <- sample(1:nrow(data_full), 2500, replace = F)

train_normal <- data_full[-val_ind, ]
valid_normal <- data_full[val_ind, ]

train_normal_scaled <- data_normal_scaled[-val_ind, ]
valid_normal_scaled <- data_normal_scaled[val_ind, ]

## Sélection de variables
vars2keep_1 <- c("as_image", "helpful_vote", "verified_purchase", "main_category",
                 "average_rating", "rating_number", "price", "categories_grp",
                 "predicted_llm_rating")

## Données pour XGB
# n = normal, s = scaled, w = weighted
X_train_n_full <- train_normal %>% select(all_of(vars2keep_1))
X_train_ns_full <- train_normal_scaled %>% select(all_of(vars2keep_1))
y_train_n <- train_normal$rating

dmtrain_n_full <- xgb.DMatrix(data = data.matrix(X_train_n_full), label = y_train_n)
dmtrain_ns_full <- xgb.DMatrix(data = data.matrix(X_train_ns_full), label = y_train_n)


X_valid_n_full <- valid_normal %>% select(all_of(vars2keep_1))
X_valid_ns_full <- valid_normal_scaled %>% select(all_of(vars2keep_1))
y_valid_n <- valid_normal$rating

dmvalid_n_full <- xgb.DMatrix(data = data.matrix(X_valid_n_full), label = y_valid_n)
dmvalid_ns_full <- xgb.DMatrix(data = data.matrix(X_valid_ns_full), label = y_valid_n)


### XGBoost --------------------------------------------------------------------
# Test boboche
xgb_temp <- xgboost(
  data = dmtrain_n_full, # the data
  nround = 5000, # max number of boosting iterations
  objective = "multi:softmax", # the objective function 
  #eval_metric = "merror", # default train eval metric
  #feval = eval_score4, # personalized train eval metric
  eta = 0.01,
  num_class = 5,
  max.depth = 5,
  subsample = 0.5,
  colsample_bytree = 0.5,
  verbose = 1
)

pred_temp <- predict(xgb_temp, dmvalid_n_full)
metrics_temp <- compute_metrics(y_valid_n, pred_temp)
#saveRDS(metrics_temp, file = "xgboost/metrics/n_n5000_e01_train20k.rds")
