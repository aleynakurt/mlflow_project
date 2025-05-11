import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from mlflow.models.signature import infer_signature
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from itertools import product
from mlflow.tracking import MlflowClient

model_configs = [
    {
        "name": "RandomForest",
        "class": RandomForestClassifier,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [10, None],
            "class_weight": ["balanced"]
        }
    },
    {
        "name": "LogisticRegression",
        "class": LogisticRegression,
        "param_grid": {
            "C": [0.1, 1.0],
            "penalty": ["l2"],
            "class_weight": ["balanced"],
            "solver": ["liblinear"],
            "max_iter": [500]
        }
    },
    {
        "name": "XGBoost",
        "class": xgb.XGBClassifier,
        "param_grid": {
            "n_estimators": [100],
            "max_depth": [5, 10],
            "learning_rate": [0.1],
            "eval_metric": ["logloss"],
            "use_label_encoder": [False]
        }
    },
    {
        "name": "GradientBoosting",
        "class": GradientBoostingClassifier,
        "param_grid": {
            "n_estimators": [100],
            "max_depth": [5, 10],
            "learning_rate": [0.1]
        }
    }
]

def prepare_data(path="/Users/aleynakurt/Desktop/mlops/Churn_Modelling.csv"):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    categorical_cols = ['Geography', 'Gender']
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                      'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"Training set shape: {X_train_transformed.shape}")
    print(f"Testing set shape: {X_test_transformed.shape}")
    
    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor

def evaluate_and_log(model, name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(model.get_params())
        mlflow.log_param("model_name", name)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": recall,
            "f1_score": f1
        })
        mlflow.set_tag("type", "ensemble_comparison")
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        print("\n" + "="*60)
        print(f"MODEL: {name}")
        print("-"*60)
        print(f" Accuracy : {acc:.4f}")
        print(f" Precision: {prec:.4f}")
        print(f" Recall   : {recall:.4f}")
        print(f" F1 Score : {f1:.4f}")
        print("="*60)


mlflow.set_experiment("Churn_Prediction")

X_train, X_test, y_train, y_test, preprocessor = prepare_data()

for config in model_configs:
    keys, values = zip(*config["param_grid"].items())
    for combination in product(*values):
        params = dict(zip(keys, combination))
        model = config["class"](**params)
        run_name = f"{config['name']}_" + "_".join(f"{k}={v}" for k, v in params.items())
        evaluate_and_log(model, run_name, X_train, X_test, y_train, y_test)

client = MlflowClient()
experiment = client.get_experiment_by_name("Churn_Prediction")
runs = client.search_runs(
    experiment.experiment_id,
    order_by=["metrics.f1_score DESC"],
    max_results=3
)
print("\nTOP 3 MODELLER")
print("="*60)
for run in runs:
    print(run.data.params["model_name"], "=> F1 Score:", f"{run.data.metrics['f1_score']:.4f}")

print("\nTOP 3 MODELS ARE ADDED TO REGISTRY")
print("="*60)

for i, run in enumerate(runs, 1):
    model_name = run.data.params["model_name"]
    run_id = run.info.run_id
    
    registered_model_name = f"{model_name}_Model"

    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=registered_model_name
    )

    print(f"{i}. Model: {registered_model_name} => Versiyon: {result.version}")
