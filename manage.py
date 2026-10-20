import os
import datetime
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix
)

from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOGGING_VERBOSITY'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
warnings.filterwarnings('ignore')


import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from keras import layers
from keras.models import Sequential
from keras.callbacks import TensorBoard


def Evaluate_Model(name, model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    metrics = {
        'Метод': name,
        'Accuracy': accuracy_score(Y_test, y_pred),
        'Precision': precision_score(Y_test, y_pred),
        'Recall': recall_score(Y_test, y_pred),
        'F1-Score': f1_score(Y_test, y_pred),
        'AUC-ROC': roc_auc_score(Y_test, y_proba)
    }

    return metrics, y_pred, y_proba


def Build_Model(hidden_units=(32, 16), dropout=(0.3, 0.2),
                activation='relu', learning_rate=1e-3):
    model = Sequential()
    model.add(layers.Dense(hidden_units[0], activation=activation,
                           input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(dropout[0]))

    if len(hidden_units) > 1:
        model.add(layers.Dense(hidden_units[1], activation=activation))
        model.add(layers.Dropout(dropout[1]))

    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("===== 1. Выбор и подготовка датасета =====")
dt = pd.read_csv("data.csv")

print("Название датасета: Kaggle — Breast Cancer Wisconsin (Diagnostic)")
print(f"Исходный размер: {dt.shape}")
print(dt.head())
print("\nИнформация о датасете")
dt.info()


print("\nПодготовка датасета: предобработка")
drop_cols = []
for c in dt.columns:
    if c.lower() == 'id' or c.startswith('Unnamed'):
        drop_cols.append(c)

dt = dt.drop(columns=drop_cols)
print(f"Удалены столбцы: {drop_cols}")

dt['diagnosis'] = dt['diagnosis'].map({'M': 0, 'B': 1})
dt = dt.rename(columns={'diagnosis': 'target'})
print("Целевая переменная закодирована: M (зло.) = 0, B (добро.) = 1")

print(f"Размер после предобработки: {dt.shape}")
print(f"Количество признаков: {dt.shape[1] - 1}")
print(f"Распределение классов:\n{dt['target'].value_counts().to_string()}")


print("\nПодготовка датасета: очистка данных")
n_missing_before = dt.isna().sum().sum()
n_duplicates_before = dt.duplicated().sum()
print("До очистки:")
print(f"\tРазмер: {dt.shape}")
print(f"\tПропуски: {n_missing_before}")
print(f"\tДубликаты: {n_duplicates_before}")

before = len(dt)
dt = dt.dropna().reset_index(drop=True)
print(f"\nУдалено строк с пропусками: {before - len(dt)}")

dt = dt.drop_duplicates(subset=None).reset_index(drop=True)
print(f"Удалено дубликатов: {before - len(dt)}")

print("\nПосле очистки:")
print(f"\tРазмер: {dt.shape}")
print(f"\tПропуски: {dt.isna().sum().sum()}")
print(f"\tДубликаты: {dt.duplicated().sum()}")

X = dt.drop('target', axis=1).values
Y = dt['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("\n===== 2. Разбиение выборки =====")
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=Y
)

print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка:  {X_test.shape}")


print("\n===== 3. Методы классификации =====")
minmax = MinMaxScaler()
X_train_positive = minmax.fit_transform(X_train)
X_test_positive = minmax.transform(X_test)


param_grid_gnb = {'var_smoothing': np.logspace(0, -9, num=20)}
grid_gnb = GridSearchCV(GaussianNB(), param_grid_gnb,
                        cv=5, scoring='f1', n_jobs=-1)
grid_gnb.fit(X_train, Y_train)
best_gnb = grid_gnb.best_estimator_
print(f"GaussianNB - лучшие параметры: {grid_gnb.best_params_}")
print(f"\tF1 (CV): {grid_gnb.best_score_}")


param_grid_mnb = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
grid_mnb = GridSearchCV(MultinomialNB(), param_grid_mnb, cv=5,
                        scoring='f1', n_jobs=-1)
grid_mnb.fit(X_train_positive, Y_train)
best_mnb = grid_mnb.best_estimator_
print(f"MultinomialNB - лучшие параметры: {grid_mnb.best_params_}")
print(f"\tF1 (CV): {grid_mnb.best_score_}")


param_grid_cnb = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
grid_cnb = GridSearchCV(ComplementNB(), param_grid_cnb, cv=5,
                        scoring='f1', n_jobs=-1)
grid_cnb.fit(X_train_positive, Y_train)
best_cnb = grid_cnb.best_estimator_
print(f"ComplementNB - лучшие параметры: {grid_cnb.best_params_}")
print(f"\tF1 (CV): {grid_cnb.best_score_}")


param_grid_bnb = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
                  'binarize': [0.0, 0.3, 0.5, 0.7]}
grid_bnb = GridSearchCV(BernoulliNB(), param_grid_bnb, cv=5,
                        scoring='f1', n_jobs=-1)
grid_bnb.fit(X_train_positive, Y_train)
best_bnb = grid_bnb.best_estimator_
print(f"BernoulliNB - лучшие параметры: {grid_bnb.best_params_}")
print(f"\tF1 (CV): {grid_bnb.best_score_}")


param_grid_dt = {
    'max_depth': [2, 3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       param_grid_dt, cv=5, scoring='f1', n_jobs=-1)
grid_dt.fit(X_train, Y_train)
best_dt = grid_dt.best_estimator_
print(f"Деревья решений - лучшие параметры: {grid_dt.best_params_}")


param_grid_lda = {
    'solver': ['svd', 'lsqr'],
    'shrinkage': [None, 'auto', 0.1, 0.5]
}
grid_lda = GridSearchCV(LinearDiscriminantAnalysis(),
                        param_grid_lda, cv=5, scoring='f1', n_jobs=-1)
grid_lda.fit(X_train, Y_train)
best_lda = grid_lda.best_estimator_
print(f"ЛДА - лучшие параметры: {grid_lda.best_params_}")


param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}
grid_svm = GridSearchCV(SVC(probability=True, random_state=42),
                        param_grid_svm, cv=5, scoring='f1', n_jobs=-1)
grid_svm.fit(X_train, Y_train)
best_svm = grid_svm.best_estimator_
print(f"Метод опорных векторов - лучшие параметры: {grid_svm.best_params_}")


param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_knn = GridSearchCV(KNeighborsClassifier(),
                        param_grid_knn, cv=5, scoring='f1', n_jobs=-1)
grid_knn.fit(X_train, Y_train)
best_knn = grid_knn.best_estimator_
print(f"Метод ближайших соседей - лучшие параметры: {grid_knn.best_params_}")


print("\n===== 4. Оценка классификаторов =====")
models_scaled = {
    'GaussianNB': best_gnb,
    'Decision Tree': best_dt,
    'LDA': best_lda,
    'SVM (RBF)': best_svm,
    'k-NN': best_knn,
}

models_positive = {
    'MultinomialNB': best_mnb,
    'ComplementNB': best_cnb,
    'BernoulliNB': best_bnb,
}

results = []
roc_data = {}

for name, model in models_scaled.items():
    metrics, Y_pred, Y_proba = Evaluate_Model(
        name, model, X_train, Y_train, X_test, Y_test
    )
    results.append(metrics)
    roc_data[name] = (Y_test, Y_proba)

for name, model in models_positive.items():
    metrics, Y_pred, Y_proba = Evaluate_Model(
        name, model, X_train_positive, Y_train, X_test_positive, Y_test
    )
    results.append(metrics)
    roc_data[name] = (Y_test, Y_proba)


all_models = {**models_scaled, **models_positive}
all_X = {}

for k in models_scaled:
    all_X[k] = X_test

for k in models_positive:
    all_X[k] = X_test_positive

dt_results = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
print(dt_results.to_string(index=False))


print("\n===== 5. Визуализация ROC-кривых =====")
print("График ROC-кривых")
plt.figure(figsize=(8, 6))
for name, (Y_true, Y_proba) in roc_data.items():
    fpr, tpr, _ = roc_curve(Y_true, Y_proba)
    auc = roc_auc_score(Y_true, Y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые классификаторов')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


print("\n===== 6. Матрицы ошибок =====")
print("График матриц ошибок")
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.ravel()

for ax, (name, model) in zip(axes, all_models.items()):
    X_use = all_X[name]
    Y_pred = model.predict(X_use)
    cm = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar=False, xticklabels=['Злокач.', 'Доброкач.'],
                yticklabels=['Злокач.', 'Доброкач.'])
    ax.set_title(name)
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Истинно')

plt.tight_layout()
plt.show()


print("\n===== 7. Нейронная сеть на TensorFlow =====")
neural_model = Build_Model()
neural_model.summary()

log_root = os.path.join(tempfile.gettempdir(), "siimo_tb_logs")
os.makedirs(log_root, exist_ok=True)

log_dir = os.path.join(log_root,
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

print(f"\nTensorBoard логи находятся в: {log_dir}")

tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

history = neural_model.fit(
    X_train, Y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[tensorboard_cb, early_stop],
    verbose=1
)

print(f"\nTensorBoard логи сохранены в: {log_dir}")


print("\n===== 8. Графики обучения нейронной сети =====")
print("Графики обучения нейронной сети")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Функция потерь')
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['accuracy'], label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_title('Точность')
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(history.history['auc'], label='Train AUC')
axes[2].plot(history.history['val_auc'], label='Val AUC')
axes[2].set_title('AUC-ROC')
axes[2].set_xlabel('Эпоха')
axes[2].set_ylabel('AUC')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()


print("\n===== 9. Оценка нейронной сети на тестовой выборке =====")
Y_pred_proba_neural = neural_model.predict(X_test).ravel()
Y_pred_neural = (Y_pred_proba_neural > 0.5).astype(int)

neural_metrics = {
    'Метод': 'Neural Network',
    'Accuracy': accuracy_score(Y_test, Y_pred_neural),
    'Precision': precision_score(Y_test, Y_pred_neural),
    'Recall': recall_score(Y_test, Y_pred_neural),
    'F1-Score': f1_score(Y_test, Y_pred_neural),
    'AUC-ROC': roc_auc_score(Y_test, Y_pred_proba_neural)
}
results.append(neural_metrics)
print("\nМетрики нейронной сети:")
print(classification_report(Y_test, Y_pred_neural,
                            target_names=[
                                'Злокачественная',
                                'Доброкачественная'
                                ]
                            ))


print("\n===== 10. Эксперименты с гиперпараметрами нейронной сети =====")
experiments = [
    {'name': 'Базовая (32-16, relu, lr=1e-3)', 'hidden': (32, 16),
     'act': 'relu', 'lr': 1e-3, 'batch': 32, 'epochs': 50},
    {'name': 'Один слой (16)', 'hidden': (16,),
     'act': 'relu', 'lr': 1e-3, 'batch': 32, 'epochs': 50},
    {'name': 'Три слоя (64-32-16)', 'hidden': (64, 32),
     'act': 'relu', 'lr': 1e-3, 'batch': 32, 'epochs': 50},
    {'name': 'Активация tanh', 'hidden': (32, 16),
     'act': 'tanh', 'lr': 1e-3, 'batch': 32, 'epochs': 50},
    {'name': 'Learning rate = 1e-2', 'hidden': (32, 16),
     'act': 'relu', 'lr': 1e-2, 'batch': 32, 'epochs': 50},
    {'name': 'Batch size = 64', 'hidden': (32, 16),
     'act': 'relu', 'lr': 1e-3, 'batch': 64, 'epochs': 50},
]

exp_results = []
for exp in experiments:
    m = Build_Model(hidden_units=exp['hidden'], activation=exp['act'],
                    learning_rate=exp['lr'])
    h = m.fit(X_train, Y_train, validation_split=0.2,
              epochs=exp['epochs'], batch_size=exp['batch'], verbose=0)

    Y_proba = m.predict(X_test, verbose=0).ravel()
    Y_p = (Y_proba > 0.5).astype(int)

    exp_results.append({
        'Эксперимент': exp['name'],
        'Accuracy': accuracy_score(Y_test, Y_p),
        'F1-Score': f1_score(Y_test, Y_p),
        'AUC-ROC': roc_auc_score(Y_test, Y_proba),
        'Val Loss (min)': min(h.history['val_loss'])
    })

dt_exp = pd.DataFrame(exp_results)
print(dt_exp.to_string(index=False))


print("\n===== 11. Итоговая сравнительная таблица =====")
dt_final = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
dt_final = dt_final.round(4)
print(dt_final.to_string(index=False))


fig, ax = plt.subplots(figsize=(12, 6))
metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
dt_plot = dt_final.set_index('Метод')[metrics_cols]
dt_plot.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Сравнение метрик качества классификаторов')
ax.set_ylabel('Значение метрики')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
