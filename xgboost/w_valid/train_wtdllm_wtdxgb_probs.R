### XGBoost sur probs --------------------------------------------------------------------
X_train2_llm_xgb_full <- train_llm_xgb %>% select(all_of(vars2keep_2))
X_train2_llm_xgb_s_full <- train_llm_xgb_scaled %>% select(all_of(vars2keep_2))
y_train2_llm_xgb <- train_llm_xgb$rating

dmtrain2_llm_xgb_full <- xgb.DMatrix(data = data.matrix(X_train2_llm_xgb_full), label = y_train_llm_xgb)
dmtrain2_llm_xgb_s_full <- xgb.DMatrix(data = data.matrix(X_train2_llm_xgb_s_full), label = y_train_llm_xgb)


X_valid2_llm_xgb_full <- valid_llm_xgb %>% select(all_of(vars2keep_2))
X_valid2_llm_xgb_s_full <- valid_llm_xgb_scaled %>% select(all_of(vars2keep_2))
y_valid2_llm_xgb <- valid_llm_xgb$rating

dmvalid2_llm_xgb_full <- xgb.DMatrix(data = data.matrix(X_valid2_llm_xgb_full), label = y_valid_llm_xgb)
dmvalid2_llm_xgb_s_full <- xgb.DMatrix(data = data.matrix(X_valid2_llm_xgb_s_full), label = y_valid_llm_xgb)

# Test boboche
xgb_llm_xgb_s_3 <- xgboost(
  data = dmtrain2_llm_xgb_s_full, # the data
  nround = 5000, # max number of boosting iterations
  objective = "multi:softprob", # the objective function 
  #eval_metric = "merror", # default train eval metric
  #feval = eval_score4, # personalized train eval metric
  eta = 0.1,
  num_class = 5,
  max.depth = 4,
  subsample = 0.5,
  colsample_bytree = 0.66,
  verbose = 1
)


prob_llm_xgb_s_3 <- predict(xgb_llm_xgb_s_3, dmvalid2_llm_xgb_s_full)
prob_llm_xgb_s_3_mat <- matrix(prob_llm_xgb_s_3, nrow = 5)
prob_llm_xgb_s_3_mat <- t(prob_llm_xgb_s_3_mat)  # Transpose to align with rows
pred_llm_xgb_s_3 <- max.col(prob_llm_xgb_s_3_mat) - 1

colnames(prob_llm_xgb_s_3_mat) <- c("prob_rating_1", "prob_rating_2", "prob_rating_3", "prob_rating_4", "prob_rating_5")
res_llm_xgb_s_3 <- bind_cols(
  tibble("rating" = y_valid_llm_xgb+1), 
  "weighted_prediction" = NA, 
  tibble("predicted_llm_rating" = pred_llm_xgb_s_3+1), 
  as_tibble(prob_llm_xgb_s_3_mat)
)

metrics_llm_xgb_s_3 <- compute_metrics(y_valid_llm_xgb, pred_llm_xgb_s_3)





test_results <- res_llm_xgb_s_3 %>% 
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

