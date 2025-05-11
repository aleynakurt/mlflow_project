## Mlflow Project
This project aims to design and implement a comprehensive machine learning (ML) system to
manage the entire lifecycle of an ML model, including experiment tracking, model training,
parameter tuning, model deployment, and monitoring. The project will utilize MLflow, an open-
source platform to manage the end-to-end machine learning lifecycle.

### Objectives:
1. Experiment Tracking: Implement and demonstrate how MLflow can be used to track different
experiments, including logging parameters, metrics, and outputs.
2. Model Training and Tuning: Develop ML models and use MLflow to log different training sessions
with varying parameters and hyperparameter tuning processes.
3. Model Deployment: Package the trained model using MLflow’s model packaging tools and deploy
it as a service for real-time or batch predictions.
4. Performance Monitoring: Set up mechanisms to monitor the deployed model's performance over
time, utilizing MLflow to track drifts in model metrics.
5. Model Registry: Utilize MLflow’s Model Registry to manage model versions and lifecycle including
stage transitions like staging and production.

To run mlflow ui:
```
mlflow server --host 127.0.0.1 --port 5000
```
