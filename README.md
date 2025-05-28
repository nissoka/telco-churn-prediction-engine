# 📞 Telco Customer Churn Prediction Engine

This project develops a Machine Learning (ML) solution to predict customer churn for a subscription-based business (inspired by a Telco customer dataset). The solution leverages Amazon Web Services (AWS) for scalable data storage, model training, and deployment, and features a simple interactive user interface.

## ❓ The Challenge / Problem Statement

In subscription-based industries like telecommunications, **customer churn (customers leaving a service)** poses a significant threat to revenue and sustained growth. Acquiring new customers is often far more expensive than retaining existing ones. Without a predictive model, companies react to churn too late, losing valuable customers, or waste resources on broad, untargeted retention efforts.

The core problem this project addresses is: **How can we proactively identify customers at high risk of churning so that targeted, timely interventions (like personalized offers or support) can be implemented to improve customer retention and maximize customer lifetime value?**

## 🎯 Project Goal

To build a Churn Prediction Engine using AWS services that:
-   Predicts whether a customer is likely to churn.
-   Uses Amazon SageMaker for robust model training and deployment.
-   Includes a simple, interactive User Interface (UI) for real-time predictions.
-   Connects to Amazon S3 for efficient data storage.
-   
## 📈 Model Performance (on Test Set)



The XGBoost model achieved the following performance on the unseen test set:

-   **Accuracy:** 0.7890
-   **Precision:** 0.6394
-   **Recall:** 0.4733
-   **F1-Score:** 0.5440
-   **ROC AUC Score:** 0.8361
