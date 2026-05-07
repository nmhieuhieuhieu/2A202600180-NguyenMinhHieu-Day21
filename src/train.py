import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

EVAL_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params: dict chứa các siêu tham số cho RandomForestClassifier
        data_path: đường dẫn đến file dữ liệu huấn luyện
        eval_path: đường dẫn đến file dữ liệu đánh giá

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá
    """

    # 1.5.1: Đọc dữ liệu huấn luyện từ data_path vào DataFrame df_train
    #   và dữ liệu đánh giá từ eval_path vào DataFrame df_eval.
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 1.5.2: Tách đặc trưng và nhãn.
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # 1.5.3: Bắt đầu một MLflow run
    with mlflow.start_run():
        # 1.5.4: Ghi nhận các siêu tham số vào MLflow.
        mlflow.log_params(params)

        # Lấy model_type từ params, mặc định là random_forest
        model_kwargs = params.copy()
        model_type = model_kwargs.pop("model_type", "random_forest")

        # 1.5.5: Khởi tạo và huấn luyện mô hình theo thuật toán.
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_kwargs, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**model_kwargs, random_state=42)
        elif model_type == "logistic_regression":
            # Logistic regression ignores n_estimators, max_depth, etc.
            # We filter out params that don't belong to it for simplicity if needed,
            # but to be safe, we'll just pass kwargs that are common or none.
            valid_lr_params = {k: v for k, v in model_kwargs.items() if k in ['C', 'penalty', 'solver', 'max_iter']}
            model = LogisticRegression(**valid_lr_params, random_state=42)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Bonus 5: Cảnh Báo Lệch Lạc Dữ Liệu
        dist = y_train.value_counts(normalize=True).to_dict()
        label_distribution = {str(k): v for k, v in dist.items()}
        print("\n--- Phân Phối Dữ Liệu Huấn Luyện (Bonus 5) ---")
        for label, pct in dist.items():
            print(f"Lớp {label}: {pct:.2%}")
            if pct < 0.10:
                print(f"WARNING: Lớp {label} chiếm ít hơn 10% tổng số mẫu! Có khả năng lệch dữ liệu.")
        print("----------------------------------------------\n")

        model.fit(X_train, y_train)

        # 1.5.6: Tính accuracy và f1_score trên tập đánh giá.
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # 1.5.7: Ghi nhận các chỉ số vào MLflow.
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # 1.5.8: Log mô hình vào MLflow artifact.
        mlflow.sklearn.log_model(model, "model")

        # 1.5.9: In kết quả ra màn hình.
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # 1.5.10: Lưu metrics ra file outputs/metrics.json.
        # Bonus 3: Báo Cáo Hiệu Suất Tự Động
        report = classification_report(y_eval, preds, digits=4)
        cm = confusion_matrix(y_eval, preds)
        
        print("\n--- Báo cáo Hiệu Suất (Bonus 3) ---")
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)
        print("-----------------------------------\n")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        # 1.5.10: Lưu metrics ra file outputs/metrics.json.
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1, "label_distribution": label_distribution}, f)

        # 1.5.11: Lưu mô hình ra file models/model.pkl.
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # 1.5.12: Trả về acc
    return acc


if __name__ == "__main__":
    # Đọc siêu tham số từ params.yaml và gọi hàm train()
    if not os.path.exists("params.yaml"):
        params = {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2}
        with open("params.yaml", "w") as f:
            yaml.dump(params, f)
    else:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
    
    train(params)
