# -------------------------
# GROUP 5 - APPENDIX CODE
# Predicting Student Dropout and Academic Success
# -------------------------

# Load libraries
library(tidyverse)
library(tree)
library(randomForest)
library(rpart)
library(rpart.plot)



# -------------------------
# 1. Data Cleaning & EDA (Ida)
# -------------------------

# Load Data
data <- read.csv("data.csv", sep = ";") 

# Convert Target to Factor
data$Target <- as.factor(data$Target)

# Handle any explicit missing values
data <- na.omit(data)

dim(data)
table(data$Target)
round(prop.table(table(data$Target)), 3)

# standardize response
data$Status <- factor(data$Target, levels = c("Dropout","Enrolled","Graduate"))
cat_cols <- c("Marital.status","Application.mode","Course","Nationality",
              "Daytime.Evening.attendance","Scholarship.holder","Debtor",
              "Mother.s.qualification","Father.s.qualification","Tuition.fees.up.to.date")

for (c in intersect(cat_cols, names(data))) data[[c]] <- factor(data[[c]])

# numeric summaries
num_vars <- intersect(c("Admission.grade","Previous.qualification..grade.",
                        "Age.at.enrollment","Curricular.units.1st.sem..grade.",
                        "Curricular.units.2nd.sem..grade."), names(data))
if (length(num_vars)>0) print(summary(data[, num_vars]))



library(ggplot2)
ggplot(data, aes(x = Status)) +
  geom_bar(aes(y = (..count..)/sum(..count..)), fill="steelblue") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  geom_text(stat="count", aes(label = scales::percent(..count../sum(..count..), accuracy = 1),
                              y = (..count..)/sum(..count..)), vjust = -0.5) +
  theme_minimal() +
  labs(title="Student Status proportions", y="Proportion", x="")


data <- data %>%
  mutate(
    CU1_approved = ifelse(!is.na(Curricular.units.1st.sem..approved.), Curricular.units.1st.sem..approved., 0),
    CU2_approved = ifelse(!is.na(Curricular.units.2nd.sem..approved.), Curricular.units.2nd.sem..approved., 0),
    first_year_approved = CU1_approved + CU2_approved,
    avg_year_grade = rowMeans(select(., any_of(c("Curricular.units.1st.sem..grade.","Curricular.units.2nd.sem..grade.","Admission.grade","Previous.qualification..grade."))), na.rm = TRUE)
  )
# quick check
data %>% group_by(Status) %>% summarise(mean_first_year_approved = mean(first_year_approved, na.rm=TRUE),
                                        median_avg_grade = median(avg_year_grade, na.rm=TRUE))




# -------------------------
# 2. Decision Tree (Azamat)
# -------------------------


library(tree)

students <- na.omit(
  data[, c(
    "Status",
    "Admission.grade",
    "Previous.qualification..grade.",
    "Age.at.enrollment",
    "Curricular.units.1st.sem..grade.",
    "Curricular.units.2nd.sem..grade.",
    "Curricular.units.1st.sem..approved.",
    "Curricular.units.2nd.sem..approved.",
    "Tuition.fees.up.to.date",
    "Scholarship.holder",
    "Debtor",
    "Daytime.evening.attendance.",
    "Marital.status"
  )]
)

tree_full <- tree(Status ~ ., data = students)
summary(tree_full)

plot(tree_full)
text(tree_full, pretty = 0)

set.seed(10)
cv_tree <- cv.tree(tree_full, FUN = prune.misclass)
cv_tree

plot(cv_tree$size, cv_tree$dev, type = "b",
     xlab = "Tree size",
     ylab = "Deviance")

best_size <- cv_tree$size[which.min(cv_tree$dev)]

tree_pruned <- prune.misclass(tree_full, best = best_size)
summary(tree_pruned)

plot(tree_pruned)
text(tree_pruned, pretty = 0)


set.seed(10)
tree_errors <- rep(0, 10)

for (i in 1:10) {
  
  train_index <- sample(1:nrow(students), size = 0.8 * nrow(students))
  train_set <- students[train_index, ]
  test_set  <- students[-train_index, ]
  
  tree_full <- tree(Status ~ ., data = train_set)
  
  cv_out <- cv.tree(tree_full, FUN = prune.misclass)
  best_size <- cv_out$size[which.min(cv_out$dev)]
  tree_pruned <- prune.misclass(tree_full, best = best_size)
  
  pred <- predict(tree_pruned, newdata = test_set, type = "class")
  conf_mat <- table(test_set$Status, pred)
  tree_errors[i] <- 1 - sum(diag(conf_mat)) / sum(conf_mat)
  
  print(paste("Iteration", i, "Test Error =", round(tree_errors[i], 4)))
}

tree_errors

mean(tree_errors)

# -------------------------
# 3. Random Forest (Kenth)
# -------------------------

rf_data <- data %>% select(-Target)

set.seed(10)
rf_errors <- numeric(10)

for(i in 1:10){
  
  train_idx <- sample(1:nrow(rf_data), size = 0.8 * nrow(rf_data))
  train_set <- rf_data[train_idx, ]
  test_set  <- rf_data[-train_idx, ]
  
  # Train Random Forest model
  rf_model <- randomForest(Status ~ ., data = train_set, importance = TRUE)
  
  # Predict on test set
  rf_pred <- predict(rf_model, newdata = test_set)
  
  # Record Prediction Error
  rf_errors[i] <- mean(rf_pred != test_set$Status)
  
  print(paste("Iteration", i, "Error:", round(rf_errors[i], 4)))
}

print(paste("Mean Random Forest Test Error:", mean(rf_errors)))
varImpPlot(rf_model, main = "Random Forest Variable Importance")
...


