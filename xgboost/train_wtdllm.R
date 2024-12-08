#### Script pour entrainer le XGBoost sur les données de revues et de produits Amazon
#### LLM entrainé sur equal weights et XGB entrainé sur données normales
options(scipen = 999) # empêcher notation scientifique
library(tidyverse)
library(xgboost)
library(scales)

source("xgboost/compute_metrics.R")

### Préparation des données ----------------------------------------------------

## Importer les données
data_0_10000_llm_raw <- read.csv("data/CCO_0_10000_wtdllm_inference_train.csv")
data_10000_15000_llm_raw <- read.csv("data/NSL_10000_15000_wtdllm_inference_train.csv")
data_15000_20000_llm_raw <- read.csv("data/NSL_15000_20000_wtdllm_inference_train.csv")
data_full_llm_raw <- bind_rows(
  data_0_10000_llm_raw,  
  data_10000_15000_llm_raw, 
  data_15000_20000_llm_raw
)

#data_full_llm_xgb_raw %>% count(rating)

## Filtrer les valeurs extremes/aberrantes
# Couper rating_number à 25K
#quantile(data_full_llm_raw$rating_number, probs = c(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 0.999, 1))
# Couper helpful_vote à 10
#quantile(data_full_llm_raw$helpful_vote, probs = c(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 0.999, 1))
# Couper le prix à 500$
#quantile(data_full_llm_raw$price, probs = c(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 0.999, 1))

data_full_llm <- data_full_llm_raw %>% 
  mutate(
    rating = rating-1, # ajustement pour algo xgboost
    predicted_llm_rating = predicted_llm_rating-1,
    verified_purchase = ifelse(verified_purchase, 1, 0)
  )

## Normaliser les variables
data_llm_scaled <- data_full_llm %>% 
  mutate(across(c(helpful_vote, average_rating, rating_number, price), ~rescale(.)))

## Train/Valid split
set.seed(12345)
val_ind <- sample(1:nrow(data_full_llm), 2500, replace = F)

train_llm <- data_full_llm[-val_ind, ]
valid_llm <- data_full_llm[val_ind, ]

train_llm_scaled <- data_llm_scaled[-val_ind, ]
valid_llm_scaled <- data_llm_scaled[val_ind, ]

## Sélection de variables
#vars2keep_1 <- c("as_image", "helpful_vote", "verified_purchase", "main_category",
#                 "average_rating", "rating_number", "price", "categories_grp",
#                 "predicted_llm_rating")
vars2keep_1 <- c("as_image", "helpful_vote", "verified_purchase", "main_category",
                 "average_rating", "rating_number", "price", "categories_grp",
                 "prob_rating_1", "prob_rating_2", "prob_rating_3", "prob_rating_4", "prob_rating_5")

## Données pour XGB
# n = normal, s = scaled, w = weighted
X_train_llm_full <- train_llm %>% select(all_of(vars2keep_1))
X_train_llm_s_full <- train_llm_scaled %>% select(all_of(vars2keep_1))
y_train_llm <- train_llm$rating

dmtrain_llm_full <- xgb.DMatrix(data = data.matrix(X_train_llm_full), label = y_train_llm)
dmtrain_llm_s_full <- xgb.DMatrix(data = data.matrix(X_train_llm_s_full), label = y_train_llm)


X_valid_llm_full <- valid_llm %>% select(all_of(vars2keep_1))
X_valid_llm_s_full <- valid_llm_scaled %>% select(all_of(vars2keep_1))
y_valid_llm <- valid_llm$rating

dmvalid_llm_full <- xgb.DMatrix(data = data.matrix(X_valid_llm_full), label = y_valid_llm)
dmvalid_llm_s_full <- xgb.DMatrix(data = data.matrix(X_valid_llm_s_full), label = y_valid_llm)


### XGBoost --------------------------------------------------------------------
# Test boboche
# Test boboche
xgb_llm_s <- xgboost(
  data = dmtrain_llm_s_full, # the data
  nround = 5000, # max number of boosting iterations
  objective = "multi:softprob", # the objective function 
  #eval_metric = "merror", # default train eval metric
  #feval = eval_score4, # personalized train eval metric
  eta = 0.05,
  num_class = 5,
  max.depth = 5,
  subsample = 0.5,
  colsample_bytree = 0.66,
  verbose = 1
)

prob_llm_s <- predict(xgb_llm_s, dmvalid_llm_s_full)
prob_llm_s_mat <- matrix(prob_llm_s, nrow = 5)
prob_llm_s_mat <- t(prob_llm_s_mat)  # Transpose to align with rows
pred_llm_s <- max.col(prob_llm_s_mat) - 1

colnames(prob_llm_s_mat) <- c("prob_rating_1", "prob_rating_2", "prob_rating_3", "prob_rating_4", "prob_rating_5")
res_llm_s <- bind_cols(
  tibble("rating" = y_valid_llm+1), 
  "weighted_prediction" = NA, 
  tibble("predicted_llm_rating" = pred_llm_s+1), 
  as_tibble(prob_llm_s_mat)
)

metrics_llm_s <- compute_metrics(y_valid_llm, pred_llm_s)





test_results <- res_llm_s %>% 
  mutate(rating = factor(as.numeric(rating)),
         predicted_llm_rating = as.factor(as.numeric(predicted_llm_rating)),
         prob_rating_1 = as.numeric(prob_rating_1),
         prob_rating_2 = as.numeric(prob_rating_2),
         prob_rating_3 = as.numeric(prob_rating_3),
         prob_rating_4 = as.numeric(prob_rating_4),
         prob_rating_5 = as.numeric(prob_rating_5)) %>% 
  mutate(across(where(is.character), as.factor))

# Générer la matrice de confusion
conf_matrix <- confusionMatrix(test_results$predicted_llm_rating, 
                               test_results$rating)

# Imprimer la matrice de confusion
conf_mat <- conf_matrix$table
confusion_df <- as.data.frame.matrix(conf_mat)
confusion_df <- confusion_df %>%
  mutate(Prediction = rownames(.)) %>%
  relocate(Prediction)

# Générer le tableau avec kable et kableExtra
kable_confusion <- confusion_df %>%
  kable("html", col.names = c("Prédiction", colnames(conf_mat)), 
        caption = "Matrice de confusion") %>%
  add_header_above(c(" " = 1, "Références" = ncol(conf_mat))) %>%
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover", "condensed")) %>%
  row_spec(0, bold = TRUE,, background = "#F2F2F2") %>%  
  column_spec(1, bold = TRUE, background = "#F2F2F2")  

# Afficher le tableau
kable_confusion

# Extraire les métriques principales
precision <- conf_matrix$byClass[, "Precision"]
recall <- conf_matrix$byClass[, "Recall"]
f1 <- conf_matrix$byClass[, "F1"]
support <- colSums(conf_matrix$table)

# Créer un dataframe pour visualiser les métriques

metrics_df <- data.frame(
  Class = rownames(conf_matrix$table),
  Precision = precision,
  Recall = recall,
  F1_Score = f1,
  Support = support
)

rownames(metrics_df) <- NULL

metrics_df <- metrics_df %>%
  mutate(
    Precision = percent(Precision, accuracy = 0.1),  # Precision en %
    Recall = percent(Recall, accuracy = 0.1),        # Recall en %
    F1_Score = percent(F1_Score, accuracy = 0.1),     # F1_Score en %
    Support = round(Support)                          # Support en entier
  )

metrics_df_transposed <- metrics_df %>%
  column_to_rownames("Class") %>%  
  t() %>%  
  as.data.frame() %>%  
  rownames_to_column("Metric")  

# Générer un tableau avec kable
kable_metrics_transposed <- metrics_df_transposed %>%
  kable("html") %>%
  add_header_above(c(" " = 1, "Classes" = ncol(conf_mat))) %>%  
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover", "condensed"))

kable_metrics_transposed



test_results_long <- test_results %>%
  select(rating, prob_rating_1, prob_rating_2, prob_rating_3, prob_rating_4, prob_rating_5) %>%
  pivot_longer(cols = starts_with("prob_rating"), 
               names_to = "Predicted_Class", 
               values_to = "Probability")

avg_probabilities <- test_results_long %>%
  group_by(rating, Predicted_Class) %>%
  summarise(Average_Probability = mean(Probability), .groups = "drop")

ggplot(avg_probabilities, aes(x = Predicted_Class, y = Average_Probability, fill = rating)) +
  geom_bar(stat = "identity", position = "dodge", show.legend = TRUE) +
  labs(x = "Classe prédite", y = "Probabilité moyenne", title = "Probabilité moyenne par classe réelle et prédite") +
  facet_wrap(~ rating, scales = "free_y") + # Facetter par classe réelle (rating)
  theme_minimal() +
  theme(legend.title = element_blank(), 
        axis.text.x = element_text(angle = 45, hjust = 1))

