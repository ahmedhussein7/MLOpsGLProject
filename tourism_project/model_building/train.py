# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-tourism-experiment")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()



# One-hot encode categorical features and scale numeric features
passthrough_features = [ "CityTier", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",]

scaled_features = ["Age","MonthlyIncome","DurationOfPitch"]
binary_nominal_features = ["TypeofContact","Gender"]
nominal_features = ["Occupation","MaritalStatus","Designation"]
ordinal_features = ["ProductPitched"]
ordinal_categories = [
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]        # ProductPitched
]

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), scaled_features),
    (OneHotEncoder(drop="if_binary", handle_unknown="ignore"), binary_nominal_features),
    (OneHotEncoder(handle_unknown="ignore"), nominal_features),
    (OrdinalEncoder(categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1), ordinal_features),
    ("passthrough", passthrough_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss")

# Small grid so the pipeline runs fast on GitHub Actions.
param_grid = {
    "xgbclassifier__n_estimators": [200, 250],
    "xgbclassifier__max_depth": [6, 8, 10],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

CLASSIFICATION_THRESHOLD = 0.50

# ---------------- Tune, log, evaluate ----------------
with mlflow.start_run():

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5,  scoring="f1", n_jobs=-1)
    # grid_search = RandomizedSearchCV(model_pipeline, param_grid, n_iter=30, cv=5,
    #                         scoring="f1", n_jobs=-1, random_state=42)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination as a nested run
    results = grid_search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

    # Log the winning parameters and settings on the parent run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
    mlflow.log_param("scale_pos_weight", round(class_weight, 3))
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)
    print(f"Best CV F1: {grid_search.best_score_:.3f}")

    # Apply the same threshold the app will use
    def predict(X):
        return (best_model.predict_proba(X)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)

    train_report = classification_report(ytrain, predict(Xtrain), output_dict=True)
    test_report = classification_report(ytest, predict(Xtest), output_dict=True)
    print("Test set performance:")
    print(classification_report(ytest, predict(Xtest)))

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    })

    # ---------------- Save model and threshold ----------------
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
