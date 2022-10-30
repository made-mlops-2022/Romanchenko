# Задача классификации

используемый датасет - https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

В `prepare_data.py` находится скрипт для предобработки данных. Категориальные признаки прогоняются через OneHotEncoder,
количественные - через StandardScaler.

Решили использовать DecisionTreeClassifier, датасет маленький, что-то сложное тут будет проблемно обучить.

В качестве метрики используем F1_score.

Конфиг для обучения - `train.yaml` - содержит путь с пребобработанными данными, путь, куда должна быть загружена итоговая модель, а также доля тестовых данных в train-test сплите.

## Как использовать:
```shell
cd src
# готовим данные для обучения
python3 prepare_data.py --config ../configs/preprocess.yaml
# обучаем модель и дампим в .pkl
python3 train_model.py --config ../configs/train.yaml
# готовим данные, чтобы потом предсказать по ним таргеты
python3 prepare_data.py --config ../configs/preprocess_test.yaml
# предсказываем, результаты записываем в файлик
python3 predict.py --config ../configs/predict.yaml
```

## Самооценка

0) В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. (1 балл)
1) В пулл-реквесте проведена самооценка, распишите по каждому пункту выполнен ли критерий или нет и на сколько баллов(частично или полностью) (1 балл)

2) Выполнено EDA, закоммитьте ноутбук в папку с ноутбуками (1 балл)

3) Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкцию по запуску (3 балла)

4) Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3 балла)

5) Проект имеет модульную структуру (2 балла)
6) Использованы логгеры (2 балла)

7) Написаны тесты на отдельные модули и на прогон обучения и predict (3 балла)

8) Для тестов генерируются синтетические данные, приближенные к реальным (2 балла)
   - можно посмотреть на библиотеки https://faker.readthedocs.io/, https://feature-forge.readthedocs.io/en/latest/
   - можно просто руками посоздавать данных, собственноручно написанными функциями.

9) Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)
10) Используются датаклассы для сущностей из конфига, а не голые dict (2 балла)

11) Напишите кастомный трансформер и протестируйте его (3 балла)
   https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

12) В проекте зафиксированы все зависимости (1 балл)
13) Настроен CI для прогона тестов, линтера на основе github actions (3 балла).
Пример с пары: https://github.com/demo-ml-cicd/ml-python-package
