##########################################################
# Project: Real Estate Valuation (Taiwan dataset, UCI)
# Statistical Learning
# Beatrice Clavarino
##########################################################

##########################################################
# 1. Libraries
##########################################################
library(readxl)
library(dplyr)
library(ggplot2)
library(reshape2)
library(cluster)

library(glmnet)
library(randomForest)
library(nnet)

# Palette
my_palette <- c("#674846", "#676767","#DBD7D2","#98817B","#960018","#BC8F8F","#FFDCE2", "#58111A")



# Reproducibility & style
set.seed(123)
options(scipen = 999)
theme_set(theme_bw())

##########################################################
# 2. Load Data
##########################################################
# Import dataset
data <- read_excel("Real estate valuation data set.xlsx")

# Rename columns
colnames(data) <- c("No", "TransactionDate", "HouseAge", 
                    "DistanceMRT", "ConvenienceStores", 
                    "Latitude", "Longitude", "PricePerUnitAreaPing")

# Remove ID
data <- data %>% select(-No)

##########################################################
# 3. Cleaning & Preprocessing
##########################################################
# Check NA, duplicates
colSums(is.na(data))
sum(duplicated(data))

# Price Normalization (Ping → m²)
data <- data %>%
  mutate(PricePerUnitArea = PricePerUnitAreaPing / 3.3)

##########################################################
# UNSUPERVISED LEARNING
##########################################################

##########################################################
# 4. Exploratory Data Analysis (EDA)
##########################################################
#Summary statistics
# Select relevant numerical variables
num_vars <- data %>%
  dplyr::select(
    PricePerUnitArea,
    HouseAge,
    DistanceMRT,
    ConvenienceStores
  )

# Compute descriptive statistics
summary_stats <- num_vars %>%
  dplyr::summarise(
    across(
      everything(),
      list(
        mean   = ~mean(.),
        median = ~median(.),
        sd     = ~sd(.),
        min    = ~min(.),
        max    = ~max(.)
      ),
      .names = "{.col}_{.fn}"
    )
  )

# summary table
summary_table <- summary_stats %>%
  tidyr::pivot_longer(
    everything(),
    names_to = c("Variable", ".value"),
    names_sep = "_"
  )

summary_table


# Histograms + Density (Price per Unit Area)
ggplot(data, aes(x = PricePerUnitArea)) +
  geom_histogram(aes(y = after_stat(density)), 
                 bins = 30, 
                 fill = my_palette[7], 
                 color = my_palette[4]) +
  geom_density(color = my_palette[5], linewidth = 1) +
  labs(title = "Distribution of House Price per Unit Area",
       x = "Price (10,000 NTD/m².)", y = "Density") +
  theme(panel.grid = element_blank())

# Histograms + Density (log Price per Unit Area)
ggplot(data, aes(x = log(PricePerUnitArea))) +
  geom_histogram(aes(y = after_stat(density)),   # <<< qui mancava
                 bins = 30, 
                 fill = my_palette[3], 
                 color = my_palette[2]) +
  geom_density(color = my_palette[5], linewidth = 1) +
  labs(title = "Distribution of log Price per Unit Area",
       x = "log Price", y = "Density") +
  theme(panel.grid = element_blank())

# Scatter con smoothing
# Price vs House Age
ggplot(data, aes(HouseAge, PricePerUnitArea)) +
  geom_point(alpha = 0.4, color = my_palette[5]) +
  geom_smooth(se = FALSE, color = my_palette[2]) +
  labs(title = "Price vs House Age", x = "Age (years)", y = "Price (NTD per m²)")  +
  theme(panel.grid = element_blank())

# Price vs Distance MRT
ggplot(data, aes(DistanceMRT, PricePerUnitArea)) +
  geom_point(alpha = 0.4, color = my_palette[2]) +
  geom_smooth(se = FALSE, color = my_palette[5]) +
  labs(title = "Price vs Distance to MRT", x = "Distance to MRT (m)", y = "Price (NTD per m²)")  +
  theme(panel.grid = element_blank())

# Price vs Convenience Stores
ggplot(data, aes(ConvenienceStores, PricePerUnitArea)) +
  geom_point(alpha = 0.5, color = my_palette[6]) +
  geom_smooth(se = FALSE, color = my_palette[8]) +
  labs(title = "Price vs Convenience Stores", x = "Stores", y = "Price (NTD per m²)")  +
  theme(panel.grid = element_blank())

# Correlazione Pearson
num_vars <- data %>% 
  select(PricePerUnitArea, HouseAge, DistanceMRT, ConvenienceStores, Latitude, Longitude)

cor_matrix <- cor(num_vars)

cor_melt <- melt(cor_matrix)

# Heatmap
ggplot(cor_melt, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  scale_fill_gradient2(
    low = my_palette[2],   
    mid = "white",    
    high = my_palette[6],  
    midpoint = 0,
    limit = c(-1, 1),
    space = "Lab",
    name = "Correlation"
  ) +
   theme(
    axis.text = element_text(color = "black"),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  ) +
  labs(title = "Correlation Matrix",
       x = "", y = "")

##########################################################
# 5. PCA (Principal Component Analysis)
##########################################################

# Select predictors for PCA (excluding target and dates)
pca_vars <- data |>
  dplyr::select(HouseAge, DistanceMRT, ConvenienceStores, Latitude, Longitude)

pca_scaled <- scale(pca_vars)

# Run PCA
pca_model <- prcomp(pca_scaled, center = FALSE, scale. = FALSE)

# Explained variance
eig_values <- pca_model$sdev^2
prop_var   <- eig_values / sum(eig_values)
cum_var    <- cumsum(prop_var)

pca_variance <- data.frame(
  PC = paste0("PC", 1:length(prop_var)),
  Eigenvalue = eig_values,
  PropVar = prop_var,
  CumVar = cum_var
)
print(dplyr::mutate(pca_variance, dplyr::across(where(is.numeric), ~ round(.x, 3))))

# Scree plot with cumulative variance
ggplot(pca_variance, aes(x = seq_along(PropVar), y = PropVar)) +
  geom_col(fill = my_palette[7], color = my_palette[4]) +
  geom_point(aes(y = CumVar), size = 2, color = my_palette[1]) +
  geom_line(aes(y = CumVar), linewidth = 0.8, color = my_palette[1]) +
  scale_x_continuous(breaks = 1:nrow(pca_variance), labels = pca_variance$PC) +
  labs(title = "Scree Plot",
       x = "Principal Components", y = "Explained Variance") +
  theme(plot.title = element_text(hjust = 0.5))  +
  theme(panel.grid = element_blank())

# Loadings
loadings <- as.data.frame(pca_model$rotation)
print(round(loadings, 3))

# Barplot of loadings for PC1 and PC2
load_long <- loadings
load_long$Variable <- rownames(load_long)
load_long <- reshape2::melt(load_long, id.vars = "Variable",
                            variable.name = "PC", value.name = "Loading")

ggplot(dplyr::filter(load_long, PC %in% c("PC1","PC2")),
       aes(x = reorder(Variable, Loading), y = Loading, fill = PC)) +
  geom_col(position = "dodge", color = my_palette[2]) +
  coord_flip() +
  scale_fill_manual(values = c(PC1 = my_palette[3], PC2 = my_palette[6])) +
  labs(title = "Variable Loadings", x = "", y = "Loading") +
  theme(plot.title = element_text(hjust = 0.5))  +
  theme(panel.grid = element_blank())

# PCA Scores
scores <- as.data.frame(pca_model$x)

##Biplot (PC1 vs PC2)
# Arrows (loadings) rescaled for the score plane
scaling_factor <- 2
arrows_df <- loadings[, c("PC1","PC2")] * scaling_factor
arrows_df$Variable <- rownames(arrows_df)

# Correlation circle (PC1-PC2)
circle <- data.frame(
  x = cos(seq(0, 2*pi, length.out = 200)),
  y = sin(seq(0, 2*pi, length.out = 200))
)
loads12 <- as.data.frame(loadings[, c("PC1","PC2")])
loads12$Variable <- rownames(loadings)

ggplot() +
  geom_path(data = circle, aes(x, y), color = "grey60") +
  geom_segment(data = loads12,
               aes(x = 0, y = 0, xend = PC1, yend = PC2, color = Variable),
               arrow = grid::arrow(length = grid::unit(0.25, "cm")),
               linewidth = 1.1) +
  coord_equal(xlim = c(-1, 1), ylim = c(-1, 1)) +
  geom_hline(yintercept = 0, linewidth = 0.3, color = "grey60") +
  geom_vline(xintercept = 0, linewidth = 0.3, color = "grey60") +
  labs(
    title = "Correlation Circle",
    x = paste0("PC1 (", round(100 * prop_var[1], 1), "%)"),
    y = paste0("PC2 (", round(100 * prop_var[2], 1), "%)"),
    color = "Variables"
  ) +
  scale_color_manual(values = my_palette[c(2, 1, 4, 6, 5)]) +
  theme(plot.title = element_text(hjust = 0.5))

# Scatter delle scores
ggplot(scores, aes(PC1, PC2)) +
  geom_point(color = my_palette[5], alpha = 0.5, size = 1.2) +
  geom_hline(yintercept = 0, linewidth = 0.3, color = "grey60") +
  geom_vline(xintercept = 0, linewidth = 0.3, color = "grey60") +
  coord_equal() +
  labs(
    title = "PCA Scores",
    x = paste0("PC1 (", round(100 * prop_var[1], 1), "%)"),
    y = paste0("PC2 (", round(100 * prop_var[2], 1), "%)")
  ) +
  theme(plot.title = element_text(hjust = 0.5))  +
  theme(panel.grid = element_blank())

##########################################################
# 6. Clustering on PCA Scores
##########################################################

# Use first 3 principal components for clustering
scores_km <- scores[, 1:3]

# Elbow method: total within-cluster sum of squares (WSS)
set.seed(123)
k_grid <- 2:6
wss <- sapply(k_grid, function(k) {
  kmeans(scores_km, centers = k, nstart = 25)$tot.withinss
})
elbow_df <- data.frame(k = k_grid, wss = wss)

ggplot(elbow_df, aes(k, wss)) +
  geom_line(linewidth = 0.8, color = my_palette[7]) +
  geom_point(size = 2, color = my_palette[5]) +
  labs(title = "Elbow Method for k-means", 
       x = "Number of clusters (k)", y = "Total WSS") +
  theme(plot.title = element_text(hjust = 0.5))  +
  theme(panel.grid = element_blank())

# Silhouette method: average silhouette width
sil_avg <- sapply(k_grid, function(k) {
  km <- kmeans(scores_km, centers = k, nstart = 25)
  ss <- silhouette(km$cluster, dist(scores_km))
  mean(ss[, "sil_width"])
})
sil_df <- data.frame(k = k_grid, silhouette = sil_avg)

ggplot(sil_df, aes(k, silhouette)) +
  geom_line(linewidth = 0.8, color = my_palette[3]) +
  geom_point(size = 2, color = my_palette[5]) +
  labs(title = "Average Silhouette Width", 
       x = "Number of clusters (k)", y = "Silhouette") +
  theme(plot.title = element_text(hjust = 0.5))  +
  theme(panel.grid = element_blank())

# # Fit k-means with chosen number of clusters ( k = 5)
# set.seed(123)
# k_opt5 <- 5
# km_fit5 <- kmeans(scores[, 1:3], centers = k_opt5, nstart = 50)
# clusters5 <- factor(km_fit5$cluster)
# 
# # Profiling the 5 clusters
# profile_df5 <- data |>
#   dplyr::mutate(Cluster = clusters5) |>
#   dplyr::group_by(Cluster) |>
#   dplyr::summarise(
#     n = dplyr::n(),
#     mean_price = mean(PricePerUnitArea),
#     mean_age = mean(HouseAge),
#     mean_mrt = mean(DistanceMRT),
#     mean_stores = mean(ConvenienceStores),
#     mean_lat = mean(Latitude),
#     mean_lon = mean(Longitude)
#   )
# 
# print(profile_df5)
# 
# # Plot PCA scores colored by 5 clusters
# ggplot(scores, aes(PC1, PC2, color = clusters5)) +
#   geom_point(alpha = 0.7) +
#   scale_color_manual(values = my_palette[5:(5 + k_opt5 - 1)]) +
#   labs(title = "K-means Clusters on PCA Scores (k = 5)",
#        x = paste0("PC1 (", round(100*prop_var[1], 1), "%)"),
#        y = paste0("PC2 (", round(100*prop_var[2], 1), "%)"),
#        color = "Cluster") +
#   theme(plot.title = element_text(hjust = 0.5))


# Fit k-means with chosen number of clusters ( k = 3)
set.seed(123)
k_opt <- 3
km_fit <- kmeans(scores_km, centers = k_opt, nstart = 50)
clusters <- factor(km_fit$cluster)

# PCA scatter plot with cluster colors
ggplot(scores, aes(PC1, PC2, color = clusters)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = my_palette[c(5, 6, 2)])+
  labs(title = paste("K-means Clusters"),
       x = paste0("PC1 (", round(100*prop_var[1], 1), "%)"),
       y = paste0("PC2 (", round(100*prop_var[2], 1), "%)"),
       color = "Cluster") +
  theme(plot.title = element_text(hjust = 0.5))  +
  theme(panel.grid = element_blank())

# Profiling clusters: average characteristics
profile_df <- data |>
  dplyr::mutate(Cluster = clusters) |>
  dplyr::group_by(Cluster) |>
  dplyr::summarise(
    n = dplyr::n(),
    mean_price = mean(PricePerUnitArea),
    mean_age = mean(HouseAge),
    mean_mrt = mean(DistanceMRT),
    mean_stores = mean(ConvenienceStores),
    mean_lat = mean(Latitude),
    mean_lon = mean(Longitude)
  )
print(profile_df)

##########################################################
# SUPERVISED LEARNING
##########################################################

##########################################################
# Utility metrics Supervised
##########################################################
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
r2   <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

##########################################################
# Utility: Store Model Performance Supervised
##########################################################
results <- data.frame(
  Model = character(),
  R2_Train = numeric(),
  RMSE_Train = numeric(),
  R2_Test = numeric(),
  RMSE_Test = numeric(),
  stringsAsFactors = FALSE
)

add_results <- function(model_name, y_train, yhat_train, y_test, yhat_test) {
  data.frame(
    Model = model_name,
    R2_Train = round(r2(y_train, yhat_train), 3),
    RMSE_Train = round(rmse(y_train, yhat_train), 3),
    R2_Test = round(r2(y_test, yhat_test), 3),
    RMSE_Test = round(rmse(y_test, yhat_test), 3)
  )
}

##########################################################
# 7. Data Preparation for Supervised Learning
##########################################################

# Target on log-scale
data <- data |>
  dplyr::mutate(LogPrice = log(PricePerUnitArea))

# Predictors
predictors <- c("HouseAge", "DistanceMRT", "ConvenienceStores", "Latitude", "Longitude")

# Modeling frame
df_ml <- data[, c("LogPrice", predictors)]

# Train/Test split (70/30)
set.seed(123)
n <- nrow(df_ml)
train_idx <- sample(seq_len(n), size = floor(0.7 * n))

train_df <- df_ml[train_idx, ]
test_df  <- df_ml[-train_idx, ]

# Scaling using only training parameters
scale_with_params <- function(train_mat, test_mat) {
  mu <- apply(train_mat, 2, mean)
  sd <- apply(train_mat, 2, sd)
  sd[sd == 0] <- 1  # avoid division by zero
  train_scaled <- sweep(sweep(train_mat, 2, mu, "-"), 2, sd, "/")
  test_scaled  <- sweep(sweep(test_mat,  2, mu, "-"), 2, sd, "/")
  list(train_scaled = train_scaled, test_scaled = test_scaled,
       center = mu, scale = sd)
}

# Raw design matrices and scaled
X_train_raw <- as.matrix(train_df[, predictors])
X_test_raw  <- as.matrix(test_df[, predictors])

sc <- scale_with_params(X_train_raw, X_test_raw)
X_train_scaled <- sc$train_scaled
X_test_scaled  <- sc$test_scaled

colnames(X_train_scaled) <- predictors
colnames(X_test_scaled)  <- predictors

# Target vectors (log-scale)
y_train <- train_df$LogPrice
y_test  <- test_df$LogPrice

# Also keep test target on price scale for reporting errors in NTD/m²
y_test_price <- exp(y_test)

##########################################################
# 8. Baseline Model: Linear Regression
##########################################################

# Fit Linear Regression on training data
lm_fit <- lm(y_train ~ ., data = as.data.frame(X_train_scaled))

# Predictions
yhat_train_lm <- predict(lm_fit, newdata = as.data.frame(X_train_scaled))
yhat_test_lm  <- predict(lm_fit, newdata = as.data.frame(X_test_scaled))

# Performance Metrics
cat("=== Linear Regression Performance ===\n")
cat(" Train Set - R²:", round(r2(y_train, yhat_train_lm), 3),
    " | RMSE:", round(rmse(y_train, yhat_train_lm), 3), "\n")
cat(" Test Set  - R²:", round(r2(y_test, yhat_test_lm), 3),
    " | RMSE:", round(rmse(y_test, yhat_test_lm), 3), "\n")

# Coefficients
cat("\n=== Model Coefficients ===\n")
print(round(coef(lm_fit), 3))

# Save performance in results table
results <- rbind(
  results,
  add_results("Linear Regression", y_train, yhat_train_lm, y_test, yhat_test_lm)
)


##########################################################
# 9. Neural Network (MLP - nnet)
##########################################################

# Fit a simple MLP with 1 hidden layer (size = 5 neurons)
set.seed(123)
nn_fit <- nnet(
  x = X_train_scaled, 
  y = y_train,
  size = 5,         # hidden neurons
  linout = TRUE,    # regression, not classification
  decay = 0.01,     # weight decay (regularization)
  maxit = 500       # maximum iterations
)

# Predictions
yhat_train_nn <- predict(nn_fit, X_train_scaled)
yhat_test_nn  <- predict(nn_fit, X_test_scaled)

# Performance metrics
cat("\n=== Neural Network (MLP) Performance ===\n")
cat(" Train Set - R²:", round(r2(y_train, yhat_train_nn), 3),
    " | RMSE:", round(rmse(y_train, yhat_train_nn), 3), "\n")
cat(" Test Set  - R²:", round(r2(y_test, yhat_test_nn), 3),
    " | RMSE:", round(rmse(y_test, yhat_test_nn), 3), "\n")

# Save results in the performance table
results <- rbind(
  results,
  add_results("Neural Network (MLP)", y_train, yhat_train_nn, y_test, yhat_test_nn)
)

##########################################################
# 11. Random Forest
##########################################################

# Fit Random Forest on raw predictors
set.seed(123)
rf_fit <- randomForest(
  x = X_train_raw, 
  y = y_train,
  ntree = 500,        
  mtry = 2,           
  importance = TRUE
)

# Predictions
yhat_train_rf <- predict(rf_fit, newdata = X_train_raw)
yhat_test_rf  <- predict(rf_fit, newdata = X_test_raw)

# Performance metrics
cat("\n=== Random Forest Performance ===\n")
cat(" Train Set - R²:", round(r2(y_train, yhat_train_rf), 3),
    " | RMSE:", round(rmse(y_train, yhat_train_rf), 3), "\n")
cat(" Test Set  - R²:", round(r2(y_test, yhat_test_rf), 3),
    " | RMSE:", round(rmse(y_test, yhat_test_rf), 3), "\n")

# Save to results table
results <- rbind(
  results,
  add_results("Random Forest", y_train, yhat_train_rf, y_test, yhat_test_rf)
)


##########################################################
# 11.2. Variable importance
##########################################################

cat("\n=== Random Forest Variable Importance ===\n")
print(importance(rf_fit))

set.seed(123)
rf_fit <- randomForest(
  x = X_train_raw, 
  y = y_train,
  ntree = 500,        
  mtry = 2,           
  importance = TRUE
)

varImpPlot(rf_fit, main = "RF Variable Importance", type = 2)

##########################################################
# 11.3. Predicted vs Actual Plot
##########################################################

# Create data frame for plotting
rf_pred_df <- data.frame(
  Actual = y_test,
  Predicted = yhat_test_rf
)

# Scatter plot with 45° reference line
ggplot(rf_pred_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = my_palette[6], alpha = 0.7, size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = my_palette[8]) +
  labs(
    title = "Predicted vs Actual",
    x = "Actual Log(Price)",
    y = "Predicted Log(Price)"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(size = 12)
  )  +
  theme(panel.grid = element_blank())

##########################################################
# 12. Model Comparison — Test RMSE
##########################################################

# Sort results by RMSE on Test Set
results_sorted <- results[order(results$RMSE_Test), ]
print(results_sorted, row.names = FALSE)


