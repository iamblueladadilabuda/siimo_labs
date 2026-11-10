# Лабораторная работа №4: MLOps. Учет экспериментов и контейнеризация

**Вариант 24:** Классификация ценовых сегментов жилья (California Housing).

## Описание задачи
Цель работы — построение воспроизводимого цикла разработки и доставки модели машинного обучения.
В рамках варианта решается задача бинарной классификации: предсказать, будет ли стоимость дома выше медианы (класс `1` / `above_median`) или ниже (класс `0` / `below_median`).

**Особенности реализации:**
* Используется датасет `sklearn.datasets.fetch_california_housing` (8 числовых признаков).
* Порог медианы (1.798) рассчитывается **только** на тренировочной выборке для предотвращения утечки данных.
* Сравниваются две модели: `LogisticRegression` и `RandomForestClassifier`.
* Используются инструменты: **MLflow** (трекинг и Registry) и **Docker** (доставка API).

## Стек технологий
* Python 3.12
* MLflow 3.5.0
* scikit-learn 1.7.2
* FastAPI, Uvicorn (REST API)
* Docker, Docker Compose

## Структура проекта
```text
mlops_lab/
├── Dockerfile               # Инструкция для сборки Docker-образа
├── compose.yaml             # Декларативный запуск сервиса
├── .dockerignore            # Исключения для Docker
├── requirements.txt         # Зависимости проекта
├── src/                     # Исходный код
│   ├── train.py             # Обучение и логирование в MLflow
│   ├── promote.py           # Назначение alias 'champion' в Registry
│   ├── export_model.py      # Скачивание модели из Registry
│   ├── evaluate.py          # Оценка на тестовой выборке
│   ├── app.py               # FastAPI приложение
│   └── check_api.py         # Smoke-тест для проверки API
└── README.md                # Этот файл
```


# Инструкция по запуску
```bash
# 1. Подготовка окружения
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Запуск MLflow Tracking Server
mkdir -p mlflow-data/artifacts
mlflow server \
  --backend-store-uri sqlite:///mlflow-data/mlflow.db \
  --artifacts-destination ./mlflow-data/artifacts \
  --serve-artifacts \
  --host 127.0.0.1 \
  --port 5000

# 3. Обучение и регистрация модели
python src/train.py --algorithm lr --c 0.1
python src/train.py --algorithm lr --c 1
python src/train.py --algorithm lr --c 10
python src/train.py --algorithm rf --trees 50 --depth 3
python src/train.py --algorithm rf --trees 100 --depth 5
python src/train.py --algorithm rf --trees 200 --depth 8

python src/train.py --algorithm rf --trees 200 --depth 8 --register
python src/train.py --algorithm lr --c 10 --register
python src/promote.py --version 4

# 4. Экспорт и тестирование модели
python src/export_model.py
python src/evaluate.py
cat test_metrics.json

# 5. Сборка и запуск API в Docker
sudo docker build -t housing-api:v1 .
sudo docker run -d --name housing-api -p 127.0.0.1:8000:8000 housing-api:v1
python src/check_api.py

# 6. Запуск через Docker Compose
sudo docker stop housing-api && sudo docker rm housing-api
sudo docker compose up -d --build
sudo docker compose ps
sudo docker compose down
```
