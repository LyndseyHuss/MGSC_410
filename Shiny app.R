# MIDSEMESTER PROJECT 410
# New Sentiment Analysis with AFINN Lexicon

# --------------------------- Libraries and Seed --------------------------
# Load necessary libraries
library(readxl)        # For reading Excel files
library(dplyr)         # For data manipulation
library(rsample)       # For data splitting
library(caret)         # For modeling
library(randomForest)  # For Random Forest model
library(tidytext)      # For text mining
library(tidyr)         # For data tidying
library(purrr)         # For functional programming
library(forcats)       # For factor manipulation
library(ggplot2)       # For plotting
library(gbm)           # For GBM model

# Set seed for reproducibility
set.seed(410)

# --------------------------- Data Import --------------------------
# Read the Excel file into R
lotwize <- read_excel("lotwize_case.xlsx")

# --------------------------- Initial Exploration ------------------
# Get the dimensions of the dataset
dim(lotwize)

# Get a glimpse of the dataset structure
glimpse(lotwize)

# Check the number of NA values in 'resoFacts/stories'
sum(is.na(lotwize$`resoFacts/stories`))

# --------------------------- Preprocessing ------------------------

# Function to convert specified columns to factors
convert_to_factors <- function(df, cols) {
  df %>% mutate(across(all_of(cols), as.factor))
}

# Columns to convert to factors
factor_cols <- c("city", "homeType", "bathrooms", "bedrooms")
lotwize <- convert_to_factors(lotwize, factor_cols)

# Function to convert specified columns to numeric
numeric_conversion <- function(df, cols) {
  df %>%
    mutate(across(all_of(cols), ~ as.numeric(as.character(.))))
}

# Convert relevant columns to numeric
lotwize <- numeric_conversion(lotwize, c("price", "yearBuilt", "latitude", "longitude"))

# --------------------------- Feature Engineering ------------------

# Calculate the age of each home
lotwize <- lotwize %>%
  filter(!is.na(yearBuilt)) %>%
  mutate(age = 2024 - yearBuilt)

# Convert specific columns to logical
logical_cols <- c("resoFacts/hasView", "resoFacts/hasSpa", "resoFacts/canRaiseHorses")
lotwize <- lotwize %>%
  mutate(across(all_of(logical_cols), ~ case_when(
    . == 'TRUE' ~ TRUE,
    . == 'FALSE' ~ FALSE,
    TRUE ~ NA
  )))

# Create the 'luxury' variable based on specified conditions
lotwize <- lotwize %>%
  mutate(
    luxury = (
      rowSums(select(., all_of(logical_cols)), na.rm = TRUE) > 0 |
        `resoFacts/garageParkingCapacity` > 2 |
        `resoFacts/fireplaces` > 1 |
        `resoFacts/stories` %in% c(2, 3, 5)
    ),
    luxury = as.factor(luxury)
  )

# --------------------------- Exploratory Data Analysis -----------
# Function to remove outliers based on IQR
remove_outliers <- function(df, feature) {
  Q1 <- quantile(df[[feature]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[feature]], 0.75, na.rm = TRUE)
  IQR_value <- IQR(df[[feature]], na.rm = TRUE)
  df %>% filter(df[[feature]] >= (Q1 - 1.5 * IQR_value) & df[[feature]] <= (Q3 + 1.5 * IQR_value))
}

# Remove outliers for important features
important_features <- c("price", "age")
lotwize <- reduce(important_features, remove_outliers, .init = lotwize)

# --------------------------- Create Unique Identifier ----------------
# Add a unique identifier to each row
lotwize <- lotwize %>%
  mutate(home_id = row_number())

# --------------------------- Sentiment Analysis Preparation --------------------
# Replace "NA" strings with actual NA values and handle them
lotwize$description <- na_if(lotwize$description, "NA")
lotwize$description[is.na(lotwize$description)] <- ""

# --------------------------- Train/Test Split ----------------------
# Perform stratified split based on 'city' to ensure representation
lotwize_split <- initial_split(lotwize, prop = 0.8, strata = city)
lotwize_train <- training(lotwize_split)
lotwize_test  <- testing(lotwize_split)

# --------------------------- Determine Top 50 Cities from Training Set ------------
# Determine the top 50 cities based on the training data
top50_cities <- lotwize_train %>%
  count(city, sort = TRUE) %>%
  slice_head(n = 50) %>%
  pull(city) %>%
  as.character()

# Inspect the top50_cities
print(top50_cities)

# --------------------------- Handle Categorical Variables --------
# Define the mapping function
handle_categorical <- function(df, top_categories) {
  df %>%
    mutate(
      city = as.character(city),  # Convert to character to prevent factor issues
      city = ifelse(city %in% top_categories, city, "Other"),  # Map to top categories or "Other"
      city = factor(city, levels = c(top_categories, "Other"))  # Convert back to factor with specified levels
    )
}

# Apply the mapping to the training set
lotwize_train_clean <- handle_categorical(lotwize_train, top50_cities)

# --------------------------- Drop Unused Factor Levels -------------------
# Apply droplevels to remove any unused levels in all factor variables
lotwize_train_clean <- lotwize_train_clean %>%
  mutate(across(where(is.factor), fct_drop))

lotwize_test_clean <- lotwize_test_clean %>%
  mutate(across(where(is.factor), fct_drop))

# --------------------------- Verify 'city' Distribution After Mapping
# Check the distribution of 'city' in the training set after mapping
lotwize_train_clean %>%
  count(city) %>%
  arrange(desc(n)) %>%
  print(n = 50)

# --------------------------- Sentiment Analysis on Training Data --------------------
# --------------------------- **Replaced Bing with AFINN Sentiment Lexicon** --------------------
# Load AFINN sentiment lexicon
afinn_sentiments <- get_sentiments("afinn")

# Tokenize and join with AFINN lexicon for training data
sentiment_scores_train <- lotwize_train_clean %>%
  select(home_id, description) %>%
  unnest_tokens(word, description) %>%
  inner_join(afinn_sentiments, by = "word") %>%
  group_by(home_id) %>%
  summarise(sentiment_score = sum(value, na.rm = TRUE))  # Sum AFINN sentiment scores

# Merge sentiment scores with the training dataset
lotwize_train_clean <- lotwize_train_clean %>%
  left_join(sentiment_scores_train, by = "home_id") %>%
  mutate(
    sentiment_score = if_else(is.na(sentiment_score), 0, sentiment_score)  # Replace NA with 0 sentiment
  )

# --------------------------- Sentiment Analysis on Test Data --------------------
# Apply the same sentiment analysis to the test data using AFINN lexicon
sentiment_scores_test <- lotwize_test_clean %>%
  select(home_id, description) %>%
  unnest_tokens(word, description) %>%
  inner_join(afinn_sentiments, by = "word") %>%
  group_by(home_id) %>%
  summarise(sentiment_score = sum(value, na.rm = TRUE))  # Sum AFINN sentiment scores

# Merge sentiment scores with the test dataset
lotwize_test_clean <- lotwize_test_clean %>%
  left_join(sentiment_scores_test, by = "home_id") %>%
  mutate(
    sentiment_score = if_else(is.na(sentiment_score), 0, sentiment_score)  # Replace NA with 0 sentiment
  )

# --------------------------- Convert Bathrooms & Bedrooms -------
# Convert 'bathrooms' and 'bedrooms' to numeric if they aren't already
convert_numeric <- function(df, cols) {
  df %>%
    mutate(across(all_of(cols), ~ as.numeric(as.character(.))))
}

# Ensure 'bathrooms' and 'bedrooms' are numeric before scaling
lotwize_train_clean <- convert_numeric(lotwize_train_clean, c("bathrooms", "bedrooms"))
lotwize_test_clean  <- convert_numeric(lotwize_test_clean, c("bathrooms", "bedrooms"))

# --------------------------- Create log_price ------------------------
# Create 'log_price' by taking the natural logarithm of 'price' in the training set
lotwize_train_clean <- lotwize_train_clean %>%
  mutate(log_price = log(price))

# Create 'log_price' in the test set as well
lotwize_test_clean <- lotwize_test_clean %>%
  mutate(log_price = log(price))

# --------------------------- Feature Selection ---------------------
# Exclude 'price' since 'log_price' is the target
selected_features <- c("city", "homeType", "bathrooms", "bedrooms", "luxury", 
                       "age", "sentiment_score", "latitude", "longitude")
lotwize_train_clean <- lotwize_train_clean %>% select(all_of(selected_features), log_price)
lotwize_test_clean  <- lotwize_test_clean %>% select(all_of(selected_features), log_price)

# --------------------------- Scaling --------------------------------
# Identify all numeric columns to scale, excluding sentiment_score
scaling_columns <- c("age", "latitude", "longitude", "bathrooms", "bedrooms")

# Create the pre-processing object using only the scaling columns from the training data
preproc <- preProcess(lotwize_train_clean[scaling_columns], method = c("center", "scale"))

# Apply the scaling to the training data
lotwize_train_clean[scaling_columns] <- predict(preproc, lotwize_train_clean[scaling_columns])

# Apply the same scaling to the test data
lotwize_test_clean[scaling_columns] <- predict(preproc, lotwize_test_clean[scaling_columns])

# --------------------------- Handle Missing Values ---------------
# Replace any remaining NA values using randomForest's na.roughfix
lotwize_train_clean <- na.roughfix(lotwize_train_clean)
lotwize_test_clean  <- na.roughfix(lotwize_test_clean)

# --------------------------- Verify Final Training Data ---------------
# Ensure 'city' has multiple categories in training set
lotwize_train_clean %>%
  count(city) %>%
  arrange(desc(n)) %>%
  print(n = 50)

# --------------------------- Handle Zero Variance Columns ----------------
# Identify zero variance columns using caret's nearZeroVar function
zero_var_indices <- nearZeroVar(lotwize_train_clean, saveMetrics = TRUE)

# Display columns with zero variance
print("Zero Variance Columns:")
print(rownames(zero_var_indices[zero_var_indices$nzv == TRUE, ]))

# Extract names of zero variance columns
zero_var_cols <- rownames(zero_var_indices[zero_var_indices$nzv == TRUE, ])

# Remove zero variance columns from training and test data
if(length(zero_var_cols) > 0){
  lotwize_train_clean <- lotwize_train_clean %>% select(-all_of(zero_var_cols))
  lotwize_test_clean <- lotwize_test_clean %>% select(-all_of(zero_var_cols))
  print("Zero variance columns have been removed from both training and test sets.")
} else {
  print("No zero variance columns found.")
}

# --------------------------- Final Droplevels and Re-validation -------------------
# Apply droplevels again after all preprocessing steps to ensure no unused levels remain
lotwize_train_clean <- lotwize_train_clean %>%
  mutate(across(where(is.factor), fct_drop))

lotwize_test_clean <- lotwize_test_clean %>%
  mutate(across(where(is.factor), fct_drop))

# Re-validate zero variance columns
zero_var_indices <- nearZeroVar(lotwize_train_clean, saveMetrics = TRUE)
print("Zero Variance Columns After Dropping Unused Levels:")
print(rownames(zero_var_indices[zero_var_indices$nzv == TRUE, ]))

# --------------------------- Hyperparameter Tuning --------------------
# Define training control with cross-validation
train_control <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

### **a. Hyperparameter Tuning for Random Forest**

# Define hyperparameter grid for Random Forest
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5))

# Train the model using caret for Random Forest with log-transformed price
rf_tuned <- train(
  log_price ~ .,  # All predictors except 'log_price'
  data = lotwize_train_clean,
  method = "rf",
  trControl = train_control,
  tuneGrid = rf_grid,
  importance = TRUE,
  ntree = 500
)

# View the best hyperparameters
print("Best Hyperparameters for Random Forest:")
print(rf_tuned$bestTune)

# Predictions and RMSE for Tuned Random Forest
rf_pred_tuned <- predict(rf_tuned, newdata = lotwize_test_clean)
rf_rmse_tuned <- sqrt(mean((lotwize_test_clean$log_price - rf_pred_tuned)^2))
print(paste("Tuned Random Forest RMSE (Log-Transformed):", round(rf_rmse_tuned, 4)))

# Feature Importance for Tuned Random Forest
rf_importance_tuned <- varImp(rf_tuned, scale = FALSE)
print("Feature Importance for Tuned Random Forest:")
print(rf_importance_tuned)

# Plot Feature Importance for Tuned Random Forest
plot(rf_importance_tuned, main = "Tuned Random Forest Feature Importance")

### **b. Hyperparameter Tuning for GBM**

# Define hyperparameter grid for GBM
gbm_grid <- expand.grid(
  n.trees = c(100, 200, 300, 400, 500),
  interaction.depth = c(3, 4, 5),
  shrinkage = c(0.01, 0.05, 0.1),
  n.minobsinnode = 10
)

# Train the model using caret for GBM with log-transformed price
gbm_tuned <- train(
  log_price ~ .,  # All predictors except 'log_price'
  data = lotwize_train_clean,
  method = "gbm",
  trControl = train_control,
  tuneGrid = gbm_grid,
  verbose = FALSE
)

# View the best hyperparameters
print("Best Hyperparameters for GBM:")
print(gbm_tuned$bestTune)

# Predictions and RMSE for Tuned GBM
gbm_pred_tuned <- predict(gbm_tuned, newdata = lotwize_test_clean)
gbm_rmse_tuned <- sqrt(mean((lotwize_test_clean$log_price - gbm_pred_tuned)^2))
print(paste("Tuned GBM RMSE (Log-Transformed):", round(gbm_rmse_tuned, 4)))

# Feature Importance for Tuned GBM
gbm_importance_tuned <- varImp(gbm_tuned, scale = FALSE)
print("Feature Importance for Tuned GBM:")
print(gbm_importance_tuned)

# Plot Feature Importance for Tuned GBM
plot(gbm_importance_tuned, main = "Tuned GBM Feature Importance")

# --------------------------- Understanding RMSE for both models -------------------------

# RANDOM FOREST
# Exponentiate predictions and actual log_price to revert to original scale
rf_pred_original <- exp(rf_pred_tuned)
actual_rf_original <- exp(lotwize_test_clean$log_price)

# Calculate RMSE on Original Scale for Random Forest
rf_rmse_original <- sqrt(mean((actual_rf_original - rf_pred_original)^2))
print(paste("Tuned Random Forest RMSE (Original Scale):", round(rf_rmse_original, 2)))

# RMSE on Original Scale With Bias Correction

# Calculate residuals on Log Scale
rf_residuals <- lotwize_test_clean$log_price - rf_pred_tuned

# Calculate variance of residuals
rf_residual_variance <- var(rf_residuals)

# Bias Correction Factor
rf_bias_correction <- exp(rf_residual_variance / 2)

# Adjusted Predictions with Bias Correction
rf_pred_adjusted <- exp(rf_pred_tuned) * rf_bias_correction

# Calculate RMSE on Original Scale with Bias Correction for Random Forest
rf_rmse_original_adjusted <- sqrt(mean((actual_rf_original - rf_pred_adjusted)^2))
print(paste("Tuned Random Forest RMSE (Original Scale, Adjusted):", round(rf_rmse_original_adjusted, 2)))

# GBM MODEL
# Exponentiate predictions and actual log_price to revert to original scale
gbm_pred_original <- exp(gbm_pred_tuned)
actual_gbm_original <- exp(lotwize_test_clean$log_price)

# Calculate RMSE on Original Scale for GBM
gbm_rmse_original <- sqrt(mean((actual_gbm_original - gbm_pred_original)^2))
print(paste("Tuned GBM RMSE (Original Scale):", round(gbm_rmse_original, 2)))

# RMSE on Original Scale With Bias Correction

# Calculate residuals on Log Scale
gbm_residuals <- lotwize_test_clean$log_price - gbm_pred_tuned

# Calculate variance of residuals
gbm_residual_variance <- var(gbm_residuals)

# Bias Correction Factor
gbm_bias_correction <- exp(gbm_residual_variance / 2)

# Adjusted Predictions with Bias Correction
gbm_pred_adjusted <- exp(gbm_pred_tuned) * gbm_bias_correction

# Calculate RMSE on Original Scale with Bias Correction for GBM
gbm_rmse_original_adjusted <- sqrt(mean((actual_gbm_original - gbm_pred_adjusted)^2))
print(paste("Tuned GBM RMSE (Original Scale, Adjusted):", round(gbm_rmse_original_adjusted, 2)))

# --------------------------- End of Script -------------------------

# Check the object names to confirm the correct model object
ls()  # This will display all objects loaded from the file

class(rf_tuned)  # Check if this is a model object
class(gbm_tuned)  # Check if this is a model object

# Rename rf_tuned to avm_model
avm_model <- rf_tuned

# Save it as avm_model.RData for use in the Shiny app
save(avm_model, file = "avm_model.RData")

# Load the renamed model file
load("avm_model.RData")  # This should load avm_model

# Load the renamed model file
load("avm_model.RData")  # Ensure that avm_model (e.g., rf_tuned or gbm_tuned) is loaded correctly


# Define the top 50 cities based on training data, for the selectInput choices
top50_cities <- c(
  "Bakersfield", "Fresno", "Long Beach", "San Francisco", "San Diego", "Anaheim",
  "Fremont", "Fontana", "Los Angeles", "Hayward", "San Mateo", "Ontario",
  "Santa Clarita", "Redondo Beach", "Fullerton", "Carlsbad", "Santa Cruz", "Clovis",
  "Salinas", "Seal Beach", "Tehachapi", "Ventura", "Oxnard", "Garden Grove",
  "Santa Barbara", "Redwood City", "Gilroy", "Daly City", "Woodland Hills", "Encinitas",
  "Poway", "Santa Clara", "Lakewood", "San Carlos", "Covina", "Orange",
  "Aptos", "San Bruno", "Rancho Palos Verdes", "Shafter", "Placentia", "Sanger",
  "La Jolla", "Cupertino", "Brea", "Canoga Park", "Monterey", "South San Francisco",
  "Foster City", "Venice", "Other"
)

# Define the UI with model-expected values for homeType
ui <- fluidPage(
  titlePanel("Automated Valuation Model (AVM)"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("city", "City", choices = top50_cities),
      numericInput("bathrooms", "Number of Bathrooms", value = 2, min = 0),
      numericInput("bedrooms", "Number of Bedrooms", value = 3, min = 0),
      numericInput("age", "Age of the Property", value = 10, min = 0),
      numericInput("latitude", "Latitude", value = 33.6846),
      numericInput("longitude", "Longitude", value = -117.8265),
      selectInput("homeType", "Home Type", choices = c(
        "Apartment" = "APARTMENT",
        "Condo" = "CONDO",
        "Unknown" = "HOME_TYPE_UNKNOWN",
        "Manufactured" = "MANUFACTURED",
        "Multi Family" = "MULTI_FAMILY",
        "Single Family" = "SINGLE_FAMILY",
        "Townhouse" = "TOWNHOUSE"
      )),
      selectInput("luxury", "Is it a luxury property?", choices = c("Yes" = TRUE, "No" = FALSE)),
      numericInput("sentiment_score", "Sentiment Score", value = 0),
      actionButton("predict", "Generate Prediction")
    ),
    
    mainPanel(
      h3("Predicted Property Price"),
      textOutput("prediction"),
      h4("Recommended Similar Properties"),
      tableOutput("recommendations"),
      plotOutput("featurePlot")
    )
  )
)

get_recommendations <- function(user_data, dataset, num_recommendations = 5) {
  # Filter dataset for similar properties based on broader criteria
  recommended_houses <- dataset %>%
    filter(
      city == user_data$city,
      abs(price - user_data$predicted_price) <= 50000  # Adjust the price range as needed
    ) %>%
    arrange(abs(price - user_data$predicted_price)) %>%
    head(num_recommendations) %>%
    select(city, price, bathrooms, bedrooms, homeType, age)  # Select specific columns to display
  
  return(recommended_houses)
}

server <- function(input, output, session) {
  observeEvent(input$predict, {
    req(input$city, input$bathrooms, input$bedrooms, input$age, input$latitude, input$longitude)
    
    # Define user_data, ensuring that 'city' and 'homeType' have the same levels as in lotwize
    user_data <- data.frame(
      city = factor(input$city, levels = levels(lotwize$city)),  # Match levels with lotwize
      bathrooms = as.numeric(input$bathrooms),
      bedrooms = as.numeric(input$bedrooms),
      age = as.numeric(input$age),
      latitude = as.numeric(input$latitude),
      longitude = as.numeric(input$longitude),
      homeType = factor(input$homeType, levels = levels(lotwize$homeType)),  # Match levels with lotwize
      luxury = factor(input$luxury, levels = c(TRUE, FALSE)),
      sentiment_score = as.numeric(input$sentiment_score)
    )
    
    tryCatch({
      # Generate prediction
      prediction <- predict(avm_model, newdata = user_data)
      predicted_price <- exp(prediction)
      output$prediction <- renderText(paste("Predicted Price: $", round(predicted_price, 2)))
      
      # Add predicted price to user_data for recommendation filtering
      user_data$predicted_price <- predicted_price
      
      # Generate recommendations
      recommendations <- get_recommendations(user_data, lotwize, num_recommendations = 5)
      output$recommendations <- renderTable(recommendations)
      
    }, error = function(e) {
      print("Error during prediction:")
      print(e)
      output$prediction <- renderText("An error occurred during prediction. Please check inputs.")
    })
  })
}


# Run the application
shinyApp(ui = ui, server = server)



