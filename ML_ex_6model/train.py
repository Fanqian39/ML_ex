# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
from xgboost import XGBClassifier
import lightgbm as lgb
import joblib
import warnings

# 环境配置
os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# 自定义模型适配层
class PatchedXGBClassifier(XGBClassifier):
    """XGBoost分类器兼容层"""

    def __init__(self, **kwargs):
        kwargs.pop("use_label_encoder", None)  # 强制移除废弃参数
        super().__init__(use_label_encoder=False, **kwargs)  # 显式禁用旧编码器

    def _more_tags(self):
        return {
            'non_deterministic': True,
            'requires_y': True,
            'X_types': ['2darray', 'sparse', '1dlabels']
        }


class PatchedLGBMClassifier(lgb.LGBMClassifier):
    """LightGBM分类器兼容层"""

    def _more_tags(self):
        return {
            'allow_nan': True,
            'non_deterministic': True,
            'requires_y': True,
            'X_types': ['2darray', 'sparse', '1dlabels']  # 补全缺失标签[[3]][[13]]
        }


# 确保权重目录存在
os.makedirs('./weights', exist_ok=True)


def load_and_preprocess(filepath):
    df = pd.read_excel(filepath, engine='openpyxl')
    X = df.iloc[:, :14]
    y = df.iloc[:, 14]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, random_state=42),
        "GBDT": GradientBoostingClassifier(
            n_estimators=100, random_state=42),
        "XGBoost": PatchedXGBClassifier(
            n_estimators=100,
            eval_metric='logloss',
            random_state=42),
        "LightGBM": PatchedLGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbosity=-1)
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess(
        r"D:\ML\ex3\智慧农业数据集.xlsx")

    results = []
    models = get_models()

    for name, model in models.items():
        start = time.time()

        try:
            cv_acc = cross_val_score(
                model, X_train, y_train, cv=5,
                scoring='accuracy', n_jobs=-1
            ).mean()
        except Exception as e:
            cv_acc = "N/A"
            print(f"{name}交叉验证异常:", str(e))

        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        duration = time.time() - start

        results.append({
            "Model": name,
            "CV Accuracy": f"{cv_acc:.4f}" if isinstance(cv_acc, float) else cv_acc,
            "Test Accuracy": f"{test_acc:.4f}",
            "Time(s)": f"{duration:.2f}"
        })

        joblib.dump(model, f"./weights/{name}.pkl")

    print("\n模型性能对比：")
    print(pd.DataFrame(results).sort_values("Test Accuracy", ascending=False).to_string(index=False))
