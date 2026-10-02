import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 1. Veriyi yükleme ve hedefi oluşturma
file_path = Path(__file__).resolve().parent / "data" / "diabetic_data.csv"

df = pd.read_csv(
    file_path,
    keep_default_na=False,
    na_values=["?"],
    low_memory=False,
)

df["readmitted_30"] = df["readmitted"].map({
    "<30": 1,
    ">30": 0,
    "NO": 0,
})

# 2. Model girdilerini ve hedefi ayırma
excluded_columns = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "readmitted",
    "readmitted_30",
]

X = df.drop(columns = excluded_columns)
y = df["readmitted_30"]

# 3. Hastalara göre eğitim-test ayrımı
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, groups=df["patient_nbr"]))

X_train = X.iloc[train_idx]         # bir tablodan veya sütundan satırları konumlarına göre seçer
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# 4. Sayısal ve kategorik sütunları belirleme
numeric_columns = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

categorical_columns = []

for column in X_train.columns:
    if column not in numeric_columns:
        categorical_columns.append(column)


# 5. Ön işleme: eksik değer, kodlama ve ölçekleme
def prepare_features(train_data,evaluation_data):

    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_train_cat = cat_imputer.fit_transform(train_data[categorical_columns])
    X_test_cat = cat_imputer.transform(evaluation_data[categorical_columns])

    cat_encoder = OneHotEncoder(handle_unknown="ignore")

    X_train_cat_encoded = cat_encoder.fit_transform(X_train_cat)
    X_test_cat_encoded = cat_encoder.transform(X_test_cat)

    num_scaler = StandardScaler()

    X_train_num_scaled = num_scaler.fit_transform(train_data[numeric_columns])
    X_test_num_scaled = num_scaler.transform(evaluation_data[numeric_columns])

    train_ready = hstack([X_train_num_scaled,X_train_cat_encoded],format="csr")
    evaluation_ready = hstack([X_test_num_scaled,X_test_cat_encoded],format = "csr")

    return train_ready,evaluation_ready


# 6. İlk modelleri eğitme ve test sonuçlarını kaydetme
X_train_ready, X_test_ready = prepare_features(X_train, X_test)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_ready,y_train)

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(X_train_ready,y_train)

y_test_pred_knn = knn_model.predict(X_test_ready)

y_test_pred_lr = lr_model.predict(X_test_ready)

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(X_train_ready,y_train)

y_test_pred_tree = tree_model.predict(X_test_ready)

model_results = []

for model_name, predictions in [
    ("Logistic Regression", y_test_pred_lr),
    ("KNN", y_test_pred_knn),
    ("Decision Tree", y_test_pred_tree),
]:
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    model_results.append({
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "TN": int(matrix[0, 0]),
        "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]),
        "TP": int(matrix[1, 1]),
    })

results_df = pd.DataFrame(model_results)

results_folder = Path(file_path).parent.parent / "results"
results_folder.mkdir(exist_ok=True)

results_df.to_csv(
    results_folder / "model_comparison.csv",
    index=False,
)

# 7. Eğitim verisi içinde hasta bazlı doğrulama
train_groups = df.iloc[train_idx]["patient_nbr"]

cv = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

# Logistic Regression doğrulaması ve karar eşiği karşılaştırması
cv_results = []
threshold_results = []

for fold_train_idx, fold_test_idx in cv.split(
    X_train,
    groups=train_groups
):
    X_fold_train = X_train.iloc[fold_train_idx]
    X_fold_test = X_train.iloc[fold_test_idx]

    y_fold_train = y_train.iloc[fold_train_idx]
    y_fold_test = y_train.iloc[fold_test_idx]

    X_fold_train_ready, X_fold_test_ready = prepare_features(X_fold_train, X_fold_test)

    cv_lr_model = LogisticRegression(max_iter=1000)

    cv_lr_model.fit(X_fold_train_ready,y_fold_train)

    y_fold_pred = cv_lr_model.predict(X_fold_test_ready)

    y_fold_proba = cv_lr_model.predict_proba(X_fold_test_ready)[:, 1]

    for threshold in [0.10, 0.20, 0.30, 0.50]:
        threshold_pred = (y_fold_proba >= threshold).astype(int)

        matrix = confusion_matrix(
            y_fold_test, threshold_pred, labels=[0, 1]
        )

        threshold_results.append({
            "split": len(cv_results) + 1,
            "threshold": threshold,
            "precision": precision_score(
                y_fold_test, threshold_pred, zero_division=0
            ),
            "recall": recall_score(
                y_fold_test, threshold_pred, zero_division=0
            ),
            "f1": f1_score(
                y_fold_test, threshold_pred, zero_division=0
            ),
            "TN": int(matrix[0, 0]),
            "FP": int(matrix[0, 1]),
            "FN": int(matrix[1, 0]),
            "TP": int(matrix[1, 1]),
        })

    matrix = confusion_matrix(y_fold_test, y_fold_pred, labels=[0, 1])

    cv_results.append({
        "split": len(cv_results) + 1,
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_fold_test, y_fold_pred),
        "precision": precision_score(y_fold_test, y_fold_pred, zero_division=0),
        "recall": recall_score(y_fold_test, y_fold_pred, zero_division=0),
        "f1": f1_score(y_fold_test, y_fold_pred, zero_division=0),
        "TN": int(matrix[0, 0]),
        "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]),
        "TP": int(matrix[1, 1]),
    })

cv_results_df = pd.DataFrame(cv_results)

cv_results_df.to_csv(
    results_folder / "logistic_regression_cv.csv",
    index=False,
)

# Her eşik için doğrulama sonuçlarını ve ortalama skorları kaydetme
threshold_results_df = pd.DataFrame(threshold_results)

threshold_results_df.to_csv(
    results_folder / "logistic_regression_threshold_cv.csv",
    index=False,
)

threshold_summary = (
    threshold_results_df
    .groupby("threshold")[["precision", "recall", "f1"]]
    .mean()
    .reset_index()
    .sort_values("f1", ascending=False)
)

threshold_summary.to_csv(
    results_folder / "logistic_regression_threshold_summary.csv",
    index=False,
)

# KNN doğrulaması
cv_results = []

for fold_train_idx, fold_test_idx in cv.split(
    X_train,
    groups=train_groups
):
    X_fold_train = X_train.iloc[fold_train_idx]
    X_fold_test = X_train.iloc[fold_test_idx]

    y_fold_train = y_train.iloc[fold_train_idx]
    y_fold_test = y_train.iloc[fold_test_idx]

    X_fold_train_ready, X_fold_test_ready = prepare_features(X_fold_train, X_fold_test)

    cv_knn_model = KNeighborsClassifier(n_neighbors=5)

    cv_knn_model.fit(X_fold_train_ready,y_fold_train)

    y_fold_pred = cv_knn_model.predict(X_fold_test_ready)

    matrix = confusion_matrix(y_fold_test, y_fold_pred, labels=[0, 1])

    cv_results.append({
        "split": len(cv_results) + 1,
        "model": "KNN",
        "accuracy": accuracy_score(y_fold_test, y_fold_pred),
        "precision": precision_score(y_fold_test, y_fold_pred, zero_division=0),
        "recall": recall_score(y_fold_test, y_fold_pred, zero_division=0),
        "f1": f1_score(y_fold_test, y_fold_pred, zero_division=0),
        "TN": int(matrix[0, 0]),
        "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]),
        "TP": int(matrix[1, 1]),
    })

cv_results_df = pd.DataFrame(cv_results)

cv_results_df.to_csv(
    results_folder / "knn_cv.csv",
    index=False,
)

# 8. Decision Tree parametrelerini karşılaştırma
param_grid = {
    "max_depth" : [5,10,None],
    "min_samples_leaf" : [1,10]
}

cv_results = []

for depth in param_grid["max_depth"]:
    for leaf in param_grid["min_samples_leaf"]:

        split_number = 0
        for fold_train_idx, fold_test_idx in cv.split(
                X_train,
                groups=train_groups
        ):
            split_number += 1
            X_fold_train = X_train.iloc[fold_train_idx]
            X_fold_test = X_train.iloc[fold_test_idx]

            y_fold_train = y_train.iloc[fold_train_idx]
            y_fold_test = y_train.iloc[fold_test_idx]

            X_fold_train_ready, X_fold_test_ready = prepare_features(X_fold_train, X_fold_test)

            grid_tree_model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=42
            )

            grid_tree_model.fit(X_fold_train_ready, y_fold_train)

            y_fold_pred = grid_tree_model.predict(X_fold_test_ready)

            matrix = confusion_matrix(y_fold_test, y_fold_pred, labels=[0, 1])

            cv_results.append({
                "split": split_number,
                "model": "Decision Tree",
                "accuracy": accuracy_score(y_fold_test, y_fold_pred),
                "precision": precision_score(y_fold_test, y_fold_pred, zero_division=0),
                "recall": recall_score(y_fold_test, y_fold_pred, zero_division=0),
                "f1": f1_score(y_fold_test, y_fold_pred, zero_division=0),
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "TN": int(matrix[0, 0]),
                "FP": int(matrix[0, 1]),
                "FN": int(matrix[1, 0]),
                "TP": int(matrix[1, 1]),
            })

cv_results_df = pd.DataFrame(cv_results)

cv_results_df.to_csv(
    results_folder / "decision_tree_grid_search_cv.csv",
    index=False,
)

# 9. Deneyleri özetleme ve başlangıç modellerinin F1 grafiğini oluşturma
grid_summary = (
    cv_results_df
    .groupby(["max_depth", "min_samples_leaf"], dropna=False)[
        ["accuracy", "precision", "recall", "f1"]
    ]
    .mean()
    .reset_index()
    .sort_values("f1", ascending=False)
)

grid_summary.to_csv(
    results_folder / "decision_tree_grid_summary.csv",
    index=False,
)

lr_cv = pd.read_csv(results_folder / "logistic_regression_cv.csv")
knn_cv = pd.read_csv(results_folder / "knn_cv.csv")
tree_cv = cv_results_df[
    cv_results_df["max_depth"].isna()
    & (cv_results_df["min_samples_leaf"] == 1)
].copy()

tree_cv.to_csv(results_folder / "decision_tree_cv.csv", index=False)

all_cv = pd.concat([lr_cv, knn_cv, tree_cv], ignore_index=True)

cv_summary = (
    all_cv.groupby("model")[["accuracy", "precision", "recall", "f1"]]
    .mean()
    .reset_index()
)

cv_summary.to_csv(results_folder / "cv_model_summary.csv", index=False)

plt.figure(figsize=(8, 5))

plt.bar(cv_summary["model"], cv_summary["f1"])

plt.title("Modellerin Ortalama CV F1 Skorları")
plt.ylabel("Ortalama F1")
plt.ylim(0, 1)

plt.savefig(
    results_folder / "cv_f1_comparison.png",
    dpi=150,
    bbox_inches="tight",
)

# 10. Doğrulamada seçilen 0.10 eşiğiyle Logistic Regression test değerlendirmesi
selected_threshold = 0.10

y_test_proba = lr_model.predict_proba(X_test_ready)[:, 1]
y_test_pred_threshold = (
    y_test_proba >= selected_threshold
).astype(int)

matrix = confusion_matrix(
    y_test, y_test_pred_threshold, labels=[0, 1]
)

final_results = pd.DataFrame([{
    "model": "Logistic Regression",
    "threshold": selected_threshold,
    "accuracy": accuracy_score(y_test, y_test_pred_threshold),
    "precision": precision_score(
        y_test, y_test_pred_threshold, zero_division=0
    ),
    "recall": recall_score(
        y_test, y_test_pred_threshold, zero_division=0
    ),
    "f1": f1_score(
        y_test, y_test_pred_threshold, zero_division=0
    ),
    "TN": int(matrix[0, 0]),
    "FP": int(matrix[0, 1]),
    "FN": int(matrix[1, 0]),
    "TP": int(matrix[1, 1]),
}])

final_results.to_csv(
    results_folder / "selected_model_test_results.csv",
    index=False,
)



