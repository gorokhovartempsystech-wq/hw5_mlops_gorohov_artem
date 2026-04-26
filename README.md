# HW5 MLOps. Воспроизводимость эксперимента

Учебный проект по курсу «Развертывание ML моделей» (МФТИ, модуль 5).
Минимальный MLOps-контур на DVC + MLflow с пайплайном из двух стадий
на датасете Iris.

## Цель проекта

Собрать воспроизводимый эксперимент: данные версионируются через DVC,
параметры вынесены в `params.yaml`, обучение оборачивается в DVC-пайплайн,
параметры, метрики и модель логируются в MLflow.

## Как запустить

```bash
git clone https://github.com/gorokhovartempsystech-wq/hw5_mlops_gorohov_artem.git
cd hw5_mlops_gorohov_artem
pip install -r requirements.txt
dvc pull
dvc repro
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

После `dvc repro` модель окажется в `model.pkl`, метрики в `metrics.json`.
MLflow UI поднимется на http://127.0.0.1:5000.

## Описание пайплайна

Пайплайн описан в `dvc.yaml` и состоит из двух стадий:

- **prepare** — `src/prepare.py` читает `data/raw/iris.csv`, делит на train
  и test по `split_ratio` из `params.yaml`, кладёт результат в
  `data/processed/`.
- **train** — `src/train.py` обучает RandomForestClassifier на train.csv,
  считает accuracy и f1 на test.csv, сохраняет модель в `model.pkl`,
  метрики в `metrics.json`. Параметры, метрики и модель логируются
  в MLflow (sqlite-бэкенд, файл `mlflow.db`).

Данные `data/raw/iris.csv` и артефакты пайплайна (`data/processed/`,
`model.pkl`) хранятся в DVC-кэше и пушатся в локальный remote
`../hw5_mlops_dvcstorage`. В git попадают только `.dvc`-файлы и
`dvc.lock` с хешами.

## Где смотреть UI MLflow

Локально:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Открыть http://127.0.0.1:5000. Эксперимент называется `hw5_iris`.

## Структура проекта

```
hw5_mlops/
├── data/
│   ├── raw/
│   │   ├── iris.csv         # под DVC, в git нет
│   │   └── iris.csv.dvc
│   └── processed/           # output стадии prepare
├── src/
│   ├── prepare.py
│   └── train.py
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```
