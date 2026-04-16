# ============================================================
# Stroke Prediction – Supervised Machine Learning Project
# Author: Islam Gamal
# Course: HarvardX Data Science Capstone (edX)
# Date: April 2026
# ============================================================

# --- 0. Package Management ----------------------------------
packages <- c("tidyverse", "caret", "randomForest", "pROC",
              "ggplot2", "gridExtra", "corrplot", "e1071",
              "ROSE", "knitr", "kableExtra")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

cat("✓ All packages loaded.\n")

# --- 1. Download Dataset ------------------------------------
# The Stroke Prediction Dataset is hosted on Kaggle.
# Since direct Kaggle API calls need credentials, I'm using
# a reliable public mirror on GitHub that mirrors Kaggle CSVs.
# This ensures reproducibility without a Kaggle account.

data_url <- paste0(
  "https://raw.githubusercontent.com/dsrscientist/",
  "dataset1/master/stroke-data.csv"
)

# Fallback: I'll also embed the direct download approach
# using a known public CSV hosting of this dataset.
data_url_fallback <- paste0(
  "https://raw.githubusercontent.com/",
  "Rohit-Gupta-Web3/Stroke-Prediction-Dataset/",
  "main/healthcare-dataset-stroke-data.csv"
)

data_file <- "stroke_data.csv"

if (!file.exists(data_file)) {
  tryCatch({
    download.file(data_url, destfile = data_file, mode = "wb")
    cat("✓ Dataset downloaded from primary URL.\n")
  }, error = function(e) {
    tryCatch({
      download.file(data_url_fallback, destfile = data_file, mode = "wb")
      cat("✓ Dataset downloaded from fallback URL.\n")
    }, error = function(e2) {
      stop("Could not download dataset. Please check internet connection.")
    })
  })
} else {
  cat("✓ Dataset already exists locally. Loading from disk.\n")
}

# --- 2. Load and Inspect ------------------------------------
stroke_raw <- read.csv(data_file, stringsAsFactors = FALSE)

cat("\n--- Dataset Dimensions ---\n")
cat("Rows:", nrow(stroke_raw), " | Columns:", ncol(stroke_raw), "\n")
cat("\n--- Column Names ---\n")
print(names(stroke_raw))
cat("\n--- Structure ---\n")
str(stroke_raw)
cat("\n--- First 6 rows ---\n")
print(head(stroke_raw))

# --- 3. Data Cleaning ---------------------------------------

# 3a. Remove the ID column (no predictive value)
stroke <- stroke_raw %>% select(-id)

# 3b. Inspect missing values
cat("\n--- Missing Values per Column ---\n")
print(colSums(is.na(stroke) | stroke == "N/A" | stroke == "Unknown"))

# 3c. 'bmi' column is read as character due to "N/A" strings
stroke$bmi <- as.numeric(stroke$bmi)

# 3d. Impute missing BMI with median (robust to outliers)
bmi_median <- median(stroke$bmi, na.rm = TRUE)
stroke$bmi[is.na(stroke$bmi)] <- bmi_median
cat(sprintf("✓ Imputed %d missing BMI values with median (%.2f)\n",
            sum(is.na(stroke_raw$bmi)), bmi_median))

# 3e. Exclude 'Unknown' in smoking_status — too ambiguous to impute
# I considered using mode imputation, but 'Unknown' is a distinct
# category representing missing data that isn't random; dropping is safer.
stroke <- stroke %>% filter(smoking_status != "Unknown")
cat(sprintf("✓ Removed rows with Unknown smoking_status. Remaining: %d\n",
            nrow(stroke)))

# 3f. Convert target and categorical variables to factors
stroke <- stroke %>%
  mutate(
    stroke          = factor(stroke, levels = c(0, 1),
                             labels = c("No", "Yes")),
    gender          = factor(gender),
    ever_married    = factor(ever_married),
    work_type       = factor(work_type),
    Residence_type  = factor(Residence_type),
    smoking_status  = factor(smoking_status),
    hypertension    = factor(hypertension, levels = c(0, 1),
                             labels = c("No", "Yes")),
    heart_disease   = factor(heart_disease, levels = c(0, 1),
                             labels = c("No", "Yes"))
  )

# 3g. Remove 'Other' gender (only 1 record — not meaningful)
stroke <- stroke %>% filter(gender != "Other")
stroke$gender <- droplevels(stroke$gender)

cat("\n--- Final cleaned dataset summary ---\n")
print(summary(stroke))

# --- 4. Exploratory Data Analysis (EDA) --------------------

# 4a. Target class distribution
cat("\n--- Stroke class distribution ---\n")
print(table(stroke$stroke))
cat(sprintf("Class imbalance ratio: 1 stroke per ~%.0f non-stroke cases\n",
            sum(stroke$stroke == "No") / sum(stroke$stroke == "Yes")))

# Plot 1: Class distribution
p1 <- ggplot(stroke, aes(x = stroke, fill = stroke)) +
  geom_bar(width = 0.5, alpha = 0.85) +
  scale_fill_manual(values = c("#4E79A7", "#F28E2B")) +
  labs(title = "Stroke Case Distribution",
       x = "Stroke Outcome", y = "Count") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

# Plot 2: Age distribution by stroke
p2 <- ggplot(stroke, aes(x = age, fill = stroke)) +
  geom_histogram(binwidth = 5, position = "identity",
                 alpha = 0.65, color = "white") +
  scale_fill_manual(values = c("#4E79A7", "#F28E2B")) +
  labs(title = "Age Distribution by Stroke Outcome",
       x = "Age", y = "Count") +
  theme_minimal(base_size = 13)

# Plot 3: Glucose level boxplot
p3 <- ggplot(stroke, aes(x = stroke, y = avg_glucose_level, fill = stroke)) +
  geom_boxplot(alpha = 0.75, outlier.shape = 21, outlier.size = 1.5) +
  scale_fill_manual(values = c("#4E79A7", "#F28E2B")) +
  labs(title = "Glucose Level vs Stroke",
       x = "Stroke", y = "Avg Glucose Level (mg/dL)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

# Plot 4: BMI boxplot
p4 <- ggplot(stroke, aes(x = stroke, y = bmi, fill = stroke)) +
  geom_boxplot(alpha = 0.75, outlier.shape = 21, outlier.size = 1.5) +
  scale_fill_manual(values = c("#4E79A7", "#F28E2B")) +
  labs(title = "BMI vs Stroke",
       x = "Stroke", y = "BMI") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

# Save EDA plots
png("eda_plots.png", width = 1200, height = 900, res = 120)
grid.arrange(p1, p2, p3, p4, ncol = 2)
dev.off()
cat("✓ EDA plots saved to eda_plots.png\n")

# Plot 5: Hypertension and heart disease proportions
p5 <- stroke %>%
  group_by(hypertension, stroke) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(hypertension) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = hypertension, y = pct, fill = stroke)) +
  geom_col(alpha = 0.85) +
  scale_fill_manual(values = c("#4E79A7", "#F28E2B")) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Stroke Rate by Hypertension Status",
       x = "Hypertension", y = "Proportion") +
  theme_minimal(base_size = 13)

# Plot 6: Smoking status
p6 <- stroke %>%
  group_by(smoking_status, stroke) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(smoking_status) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = smoking_status, y = pct, fill = stroke)) +
  geom_col(alpha = 0.85) +
  scale_fill_manual(values = c("#4E79A7", "#F28E2B")) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Stroke Rate by Smoking Status",
       x = "Smoking Status", y = "Proportion") +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

png("eda_categorical.png", width = 1200, height = 500, res = 120)
grid.arrange(p5, p6, ncol = 2)
dev.off()
cat("✓ Categorical EDA plots saved.\n")

# --- 5. Feature Engineering ---------------------------------

# Create age groups (clinical relevance)
stroke <- stroke %>%
  mutate(age_group = cut(age,
                         breaks = c(0, 40, 60, 80, Inf),
                         labels = c("Young", "Middle", "Senior", "Elderly"),
                         right = FALSE))

# High glucose flag (clinical threshold > 126 mg/dL = diabetic range)
stroke <- stroke %>%
  mutate(high_glucose = factor(ifelse(avg_glucose_level > 126, 1, 0),
                               labels = c("Normal", "High")))

cat("✓ Feature engineering complete: age_group, high_glucose created.\n")

# --- 6. Train / Test Split ----------------------------------
set.seed(2024)  # reproducibility

# Stratified split (important given class imbalance)
train_idx <- createDataPartition(stroke$stroke, p = 0.80, list = FALSE)
train_data <- stroke[train_idx, ]
test_data  <- stroke[-train_idx, ]

cat(sprintf("\n✓ Train set: %d rows | Test set: %d rows\n",
            nrow(train_data), nrow(test_data)))
cat("Train stroke prevalence:", round(mean(train_data$stroke == "Yes") * 100, 2), "%\n")
cat("Test  stroke prevalence:", round(mean(test_data$stroke  == "Yes") * 100, 2), "%\n")

# --- 7. Handle Class Imbalance (ROSE oversampling) ----------
# The dataset is heavily imbalanced (~5% stroke cases).
# Without addressing this, models would simply predict 'No' always.
# I chose ROSE (Random Over-Sampling Examples) over simple SMOTE
# because it generates synthetic samples using kernel density estimation,
# which tends to produce more varied examples.

set.seed(2024)
train_balanced <- ROSE(stroke ~ ., data = train_data, seed = 2024)$data

cat(sprintf("✓ After ROSE balancing: %d rows\n", nrow(train_balanced)))
cat("Balanced stroke distribution:\n")
print(table(train_balanced$stroke))

# --- 8. Cross-Validation Control ----------------------------
ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# --- 9. Model 1: Logistic Regression (Baseline) -------------
cat("\n--- Training Logistic Regression ---\n")
set.seed(2024)

lr_model <- train(
  stroke ~ age + hypertension + heart_disease + avg_glucose_level +
           bmi + smoking_status + gender + work_type + high_glucose +
           age_group + ever_married,
  data      = train_balanced,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl,
  metric    = "ROC"
)

cat("✓ Logistic Regression trained.\n")
print(lr_model)

# --- 10. Model 2: Random Forest (Advanced) ------------------
# Random Forest is chosen because:
# 1) It handles non-linear relationships and feature interactions
# 2) Robust to outliers and doesn't require feature scaling
# 3) Built-in feature importance
# 4) Performs well on tabular healthcare data (shown in literature)
cat("\n--- Training Random Forest ---\n")
set.seed(2024)

rf_grid <- expand.grid(mtry = c(3, 5, 7))

rf_model <- train(
  stroke ~ age + hypertension + heart_disease + avg_glucose_level +
           bmi + smoking_status + gender + work_type + high_glucose +
           age_group + ever_married,
  data      = train_balanced,
  method    = "rf",
  trControl = ctrl,
  tuneGrid  = rf_grid,
  metric    = "ROC",
  ntree     = 300
)

cat("✓ Random Forest trained.\n")
print(rf_model)

# --- 11. Predictions & Evaluation --------------------------
# Logistic Regression
lr_probs <- predict(lr_model, newdata = test_data, type = "prob")[, "Yes"]
lr_preds <- predict(lr_model, newdata = test_data)
lr_cm    <- confusionMatrix(lr_preds, test_data$stroke, positive = "Yes")
lr_roc   <- roc(response  = ifelse(test_data$stroke == "Yes", 1, 0),
                predictor = lr_probs, quiet = TRUE)

# Random Forest
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[, "Yes"]
rf_preds <- predict(rf_model, newdata = test_data)
rf_cm    <- confusionMatrix(rf_preds, test_data$stroke, positive = "Yes")
rf_roc   <- roc(response  = ifelse(test_data$stroke == "Yes", 1, 0),
                predictor = rf_probs, quiet = TRUE)

cat("\n===== LOGISTIC REGRESSION RESULTS =====\n")
print(lr_cm)
cat(sprintf("AUC: %.4f\n", auc(lr_roc)))

cat("\n===== RANDOM FOREST RESULTS =====\n")
print(rf_cm)
cat(sprintf("AUC: %.4f\n", auc(rf_roc)))

# --- 12. Comparison Table -----------------------------------
results_table <- data.frame(
  Model    = c("Logistic Regression", "Random Forest"),
  Accuracy = c(round(lr_cm$overall["Accuracy"], 4),
               round(rf_cm$overall["Accuracy"], 4)),
  Sensitivity = c(round(lr_cm$byClass["Sensitivity"], 4),
                  round(rf_cm$byClass["Sensitivity"], 4)),
  Specificity = c(round(lr_cm$byClass["Specificity"], 4),
                  round(rf_cm$byClass["Specificity"], 4)),
  AUC      = c(round(auc(lr_roc), 4),
               round(auc(rf_roc), 4))
)

cat("\n===== MODEL COMPARISON TABLE =====\n")
print(results_table)

# --- 13. ROC Curve Plot -------------------------------------
png("roc_curves.png", width = 900, height = 700, res = 120)
plot(lr_roc, col = "#4E79A7", lwd = 2.5,
     main = "ROC Curves: Logistic Regression vs Random Forest")
plot(rf_roc, col = "#F28E2B", lwd = 2.5, add = TRUE)
legend("bottomright",
       legend = c(sprintf("Logistic Regression (AUC = %.3f)", auc(lr_roc)),
                  sprintf("Random Forest (AUC = %.3f)",       auc(rf_roc))),
       col    = c("#4E79A7", "#F28E2B"),
       lwd    = 2.5, bty = "n")
dev.off()
cat("✓ ROC curve saved to roc_curves.png\n")

# --- 14. Feature Importance (RF) ----------------------------
rf_imp <- varImp(rf_model)$importance
rf_imp$Feature <- rownames(rf_imp)

p_imp <- ggplot(rf_imp %>% arrange(desc(Overall)) %>% head(12),
                aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "#F28E2B", alpha = 0.85) +
  coord_flip() +
  labs(title = "Random Forest – Top 12 Feature Importances",
       x = "Feature", y = "Importance (Mean Decrease Gini)") +
  theme_minimal(base_size = 13)

png("feature_importance.png", width = 900, height = 600, res = 120)
print(p_imp)
dev.off()
cat("✓ Feature importance plot saved.\n")

# --- 15. Save Results ---------------------------------------
write.csv(results_table, "model_comparison.csv", row.names = FALSE)
saveRDS(rf_model, "rf_model.rds")
saveRDS(lr_model, "lr_model.rds")
cat("\n✓ Models and results saved to disk.\n")
cat("\n========== SCRIPT COMPLETE ==========\n")
