#### Script pour entrainer le XGBoost sur les données de revues et de produits Amazon
#### LLM entrainé sur equal weights avec custom loss et polr entrainé sur equal weights
options(scipen = 999) # empêcher notation scientifique
library(tidyverse)
#library(MASS)
library(scales)

source("xgboost/compute_metrics.R")

### Préparation des données ----------------------------------------------------

## Importer les données
data_0_20000_llm_xgb__customloss_raw <- read.csv("data/CCO_0_20000_wtdllm_inference_wtdtrain_customloss.csv")
data_full_llm_xgb_closs_raw <- data_0_20000_llm_xgb__customloss_raw

data_0_10000_llm_raw <- read.csv("data/CCO_0_10000_wtdllm_inference_train.csv")
data_10000_15000_llm_raw <- read.csv("data/NSL_10000_15000_wtdllm_inference_train.csv")
data_15000_20000_llm_raw <- read.csv("data/NSL_15000_20000_wtdllm_inference_train.csv")
data_full_llm_raw <- bind_rows(
  data_0_10000_llm_raw,  
  data_10000_15000_llm_raw, 
  data_15000_20000_llm_raw
)

#data_full_llm_xgb_closs_raw %>% count(rating)
#data_full_llm_raw %>% count(rating)

## Filtrer les valeurs extremes/aberrantes
# Couper rating_number à 25K
#quantile(data_0_20000_llm_xgb__customloss_raw$rating_number, probs = c(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 0.999, 1))
# Couper helpful_vote à 10
#quantile(data_0_20000_llm_xgb__customloss_raw$helpful_vote, probs = c(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 0.999, 1))
# Couper le prix à 500$
#quantile(data_0_20000_llm_xgb__customloss_raw$price, probs = c(0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 0.999, 1))

data_full_polr_cl <- data_0_20000_llm_xgb__customloss_raw %>% 
  mutate(
    rating = as.factor(rating-1), # ajustement pour algo xgboost
    predicted_llm_rating = predicted_llm_rating-1,
    verified_purchase = ifelse(verified_purchase, 1, 0)
  )

data_full_polr_valid <- data_full_llm_raw %>% 
  mutate(
    rating = as.factor(rating-1), # ajustement pour algo xgboost
    predicted_llm_rating = predicted_llm_rating-1,
    verified_purchase = ifelse(verified_purchase, 1, 0)
  )

## Normaliser les variables
data_full_polr_cl_s <- data_full_polr_cl %>% 
  mutate(across(c(helpful_vote, average_rating, rating_number, price), ~rescale(.)))

data_full_polr_valid_s <- data_full_polr_valid %>% 
  mutate(across(c(helpful_vote, average_rating, rating_number, price), ~rescale(.)))


## Sélection de variables
vars2keep_1 <- c("as_image", "helpful_vote", "verified_purchase", "main_category",
                 "average_rating", "rating_number", "price", "categories_grp",
                 "predicted_llm_rating")
vars2keep_2 <- c("as_image", "helpful_vote", "verified_purchase", "main_category",
                 "average_rating", "rating_number", "price", "categories_grp",
                 "prob_rating_1", "prob_rating_2", "prob_rating_3", "prob_rating_4") #, "prob_rating_5"

vars2keep_0 <- c("as_image", "helpful_vote", "verified_purchase", "main_category",
                 "average_rating", "rating_number", "price", #"categories_grp"),
                 "prob_rating_1", "prob_rating_2", "prob_rating_3", "prob_rating_4", "prob_rating_5")

## Données pour polr
X_train_polr_cl <- data_full_polr_cl_s %>% select(c(rating, all_of(vars2keep_0))) #all_of(vars2keep_1)
y_train_polr_cl <- data_full_polr_cl_s$rating

X_valid_polr <- data_full_polr_valid_s %>% select(c(rating, all_of(vars2keep_0)))
y_valid_polr <- as.numeric(as.character(data_full_polr_valid_s$rating))



### POLR sur prob --------------------------------------------------------------------
test_polr_cl <- MASS::polr(
  formula = rating ~ .,
  data = X_train_polr_cl,
  method = "logistic", #"logistic"
  #Hess = TRUE
)

res_test_polr_cl <- predict(test_polr_cl, newdata = X_valid_polr, type = "p") #X_valid_polr

#prob_polr_mat <- matrix(res_test_polr, nrow = 5)
#prob_polr_mat <- t(prob_polr_mat)  # Transpose to align with rows
pred_polr_cl <- max.col(res_test_polr_cl) - 1

colnames(res_test_polr_cl) <- c("prob_rating_1", "prob_rating_2", "prob_rating_3", "prob_rating_4", "prob_rating_5")
res_polr_cl <- bind_cols(
  tibble("rating" = y_valid_polr+1), 
  "weighted_prediction" = NA, 
  tibble("predicted_llm_rating" = pred_polr_cl+1), 
  as_tibble(res_test_polr_cl)
)

metrics_polr_cl <- compute_metrics(y_valid_polr, pred_polr_cl)





test_results <- res_polr_cl %>% 
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

