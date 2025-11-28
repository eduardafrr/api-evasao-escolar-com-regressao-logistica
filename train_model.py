import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

RANDOM_STATE = 42
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
DATA_PATH = os.path.join("data", "alunos.csv")

def generate_synthetic(n=2000, seed=RANDOM_STATE):
    np.random.seed(seed)

    df = pd.DataFrame({
        "idade": np.random.randint(17, 36, size=n),
        "sexo": np.random.choice(["M", "F"], size=n, p=[0.45, 0.55]),
        "tipo_escola_medio": np.random.choice(["publica", "privada"], size=n, p=[0.7, 0.3]),

        "nota_enem": np.clip(np.random.normal(600, 100, n), 200, 1000),
        "cra_1_sem": np.clip(np.random.normal(6.0, 1.5, n), 0, 10),
        "reprovacoes_1_sem": np.random.poisson(0.4, n),

        "renda_familiar": np.clip(np.random.exponential(2.5, n), 0.1, 20.0),
        "bolsista": np.random.choice([0, 1], size=n, p=[0.75, 0.25]),

        "trabalha": np.random.choice([0, 1], size=n, p=[0.55, 0.45]),
        "distancia_campus_km": np.clip(np.random.exponential(7, n), 0, 80),
    })

    df["horas_trabalho_semana"] = df.apply(
        lambda row: np.random.randint(10, 45) if row["trabalha"] == 1 else 0,
        axis=1
    )

    p = (
        -0.55 * (df["cra_1_sem"] - 5) +
        0.45 * df["reprovacoes_1_sem"] +

        -0.20 * df["renda_familiar"] +
        -0.25 * df["bolsista"] +

        0.40 * df["trabalha"] +
        0.015 * df["horas_trabalho_semana"] +

        0.03 * df["distancia_campus_km"] +

        0.60 * (
            (df["cra_1_sem"] < 5).astype(int)
            * (df["trabalha"] == 1).astype(int)
        ) +
        0.45 * (
            (df["distancia_campus_km"] > 25).astype(int)
            * (df["renda_familiar"] < 2).astype(int)
        ) +

        np.random.normal(0, 0.05, n)
    )

    p = 1 / (1 + np.exp(-p * 0.8))

    df["evasao_ate_1ano"] = (np.random.rand(n) < p).astype(int)

    return df

def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Carregando dataset em {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("Dataset não encontrado em data/alunos.csv — gerando dataset sintético.")
        df = generate_synthetic(2000)
    return df

def preprocess_and_train(df):
    target = "evasao_ate_1ano"
    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = ["idade", "nota_enem", "renda_familiar", "horas_trabalho_semana", "cra_1_sem", "reprovacoes_1_sem", "distancia_campus_km"]
    categorical_features = ["sexo", "tipo_escola_medio", "trabalha", "bolsista"]  # trabalhar/bools como categórico aceitável

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }
    print("Métricas:", metrics)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(os.path.join(MODEL_DIR, "roc_curve.png"))
    plt.close()

    joblib.dump(clf, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")

    return clf, metrics

if __name__ == "__main__":
    df = load_data()
    clf, metrics = preprocess_and_train(df)
