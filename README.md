Архитектура проекта

geometric-regularization/

├── pyproject.toml          # Зависимости, сборка (Poetry или Setuptools)

├── README.md               # Описание проекта, инструкция по запуску

├── .env                    # Переменные окружения и пути

├── src/

│   ├── dataset/            # Подготовка и загрузка данных

│   │   ├── datasets.py         # Dataset-классы и описание входных данных

│   │   └── loaders.py          # DataLoader, collate-функции

│   ├── utils/              # Общие утилиты

│   │   ├── config.py           # Работа с конфигами (Hydra/OmegaConf)

│   │   ├── io.py               # Работа с файлами, путями

│   │   ├── logging.py          # Логгеры (файловый, консоль, MLflow/Weights&Biases)

│   │   ├── utils.py            # Seed-функции, таймеры, контроль повторяемости

│   │   └── registry.py         # Регистрируемые классы (модели, лоссы и т.п.)

│   ├── preprocess/         # Трансформации данных

│   │   ├── transforms.py       # Torchvision-совместимые трансформы

│   │   └── augmentations.py    # Аугментации (RandAugment, CutMix и др.)

│   ├── models/             # Архитектуры

│   │   ├── cnns.py              # Простые сверточные классификаторы

│   │   ├── transformers.py      # Варианты Vision Transformer и др.

│   │   └── head_factory.py      # Генерация classification heads (linear/MLP)

│       └── train_model.py       # Фабрика тренировки модели + инициализация

│   ├── runners/            # Основной пайплайн обучения

│   │   ├── run_clf_base.py      # Основной скрипт train/val/test цикла

│   │   └── hooks.py             # Коллбэки: логгеры, early stopping, checkpoints

│   ├── metrics/            # Метрики

│   │   ├── task.py              # Классические метрики качества: accuracy, AUROC и т.п.

│   │   ├── geometry.py          # Геометрические метрики: локальная размерность, геодезика

│   │   └── topology.py          # Метрики на основе persistent homology

│   ├── regularizers/       # Регуляризаторы

│   │   ├── geometric_loss.py    # Compactness, margin, complexity loss

│   │   └── baseline.py          # L2, Jacobian norm, spectral norm и т.п.

│   ├── vizualisation/      # Визуализация

│   │   └── vizualisator.py      # UMAP, спектры Лапласиана, графы и проекции

│   └── evaluation/         # Анализ результатов

│       ├── eval_latents.py      # Анализ скрытых представлений

│       └── sanity_check.py      # Простые sanity-check метрики (кластеризация, размерность)

├── configs/                # Конфигурации экспериментов

│   └── defaults.yaml           # Главный конфиг (dataset, модель, лоссы, логгеры)

├── experiments_result/     # Логи и артефакты экспериментов

│   ├── exp_synthetic_2g/       # Примеры: смесь гауссиан

│   ├── exp_mnist_baseline/     # Базовая модель без геометрии

│   └── exp_mnist_geo/          # Модель с геометрической регуляризацией

├── assets/                 # Фигуры, графики, визуализации, сохранённые графы

├── notebooks/              # Jupyter Notebook'и для экспериментов

│   └── prototyping.ipynb       # Прототипирование метрик и визуализаций

└── tests/                  # Модульные тесты

   ├── test_metrics.py

   ├── test_knn_graph.py

   └── test_regularizers.py
