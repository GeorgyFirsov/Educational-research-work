from os.path import exists
from itertools import chain
from typing import (Iterable,
                    List,
                    Sized,
                    Optional,
                    Union,
                    Type,
                    )

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import pickle


class DecompositionType(object):
    """
    Класс, представляющий собой модель декомпозиции временного ряда.
    Экспортирует строковое имя модели и способ комбинирования тренда,
    сезонности и остатка (сложение или умножение)
    """

    internal_mapping = {
        # Название  |  Единица | Функция комбинирования
        'additive':       (0, lambda t, s, r: t + s + r),
        'multiplicative': (1, lambda t, s, r: t * s * r)
    }

    def __init__(self, decomposition_type: str):
        if decomposition_type not in DecompositionType.internal_mapping.keys():
            raise ValueError(f'decomposition_type must be one of {DecompositionType.internal_mapping.keys()}')

        self.unit = DecompositionType.internal_mapping[decomposition_type][0]
        self.operation = DecompositionType.internal_mapping[decomposition_type][1]
        self.decomposition_type = decomposition_type

    def evaluate(self,
                 t: Optional[float] = None,
                 s: Optional[float] = None,
                 r: Optional[float] = None) -> float:
        """
        Комбинирует значения тренда, сезонности и остатка. Чтобы не учитывать
        какую-либо компоненту, надо передать None (по-умолчанию передается).
        None будет заменяться на единичный элемент относительно операции.

        :param t: трендовая составляющая
        :param s: сезонная составляющая
        :param r: остаток
        :return: комбинированное значение в соответствии с моделью
        """
        t = self.unit if t is None else t
        s = self.unit if s is None else s
        r = self.unit if r is None else r
        return self.operation(t, s, r)

    @property
    def type(self):
        return self.decomposition_type


def exponential_smoothing(series: Union[Sized, pd.DataFrame], alpha: int = 0.3) -> np.ndarray:
    """
    Предоставляет алгоритм экспоненциального сглаживания,
    задаваемый формулой:
            y_{t}n = a * y_{t} + (1 - a) * y_{t - 1}

    :param series: временной ряд
    :param alpha: параметр альфа экспоненциального сглаживания
    :return: массив со значениями, рассчитанными по методу экспоненциального сглаживания
    """

    if alpha < 0 or alpha > 1:
        raise ValueError(f'\'alpha\' parameter should be enclosed in range [0, 1], but has value of {alpha}')

    if isinstance(series, pd.DataFrame):
        series = series.values

    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])

    return np.array(result)


def double_exponential_smoothing(series: Union[Sized, pd.DataFrame], alpha: int = 0.3, beta: int = 0.3) -> np.ndarray:
    """
    Представляет алгоритм двойного экспоненциального сглаживания.
    Формуля будут выглядеть некрасиво, поэтому не привожу их.

    :param series: временной ряд
    :param alpha: параметр альфа двойного экспоненциального сглаживания
    :param beta: параметр бета двойного экспоненциального сглаживания
    :return: массив со значениями, рассчитанными по методу двойного экспоненциального сглаживания
    """

    if alpha < 0 or alpha > 1:
        raise ValueError(f'\'alpha\' parameter should be enclosed in range [0, 1], but has value of {alpha}')

    if beta < 0 or beta > 1:
        raise ValueError(f'\'beta\' parameter should be enclosed in range [0, 1], but has value of {beta}')

    if not isinstance(series, list):
        series = series.values

    result = [series[0]]
    for n in range(1, len(series) + 1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):  # прогнозируем
            value = result[-1]
        else:
            value = series[n]

        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)

    return np.array(result)


def stable_seasoal_filter(time_series: Sized, freq: int):
    """
    Стабильный сезонный фильтр для ряда.

    :param time_series: временной ряд
    :param freq: частота расчета среднего значения
    :return: значения сезонной составляющей
    """
    length = len(time_series)
    if length < freq:
        raise ValueError(f'Length of time series is less than freq ({length} < {freq}')

    if not isinstance(freq, int):
        raise TypeError(f'freq must be an integer')

    if freq < 1:
        raise ValueError(f'freq must be greater than zero (actually is {freq})')

    values = time_series.values if isinstance(time_series, pd.DataFrame) else time_series

    seasonal = list()
    for i in range(freq):
        seasonal_values = [values[i + j * freq] for j in range(length) if i + j * freq < length]
        seasonal.append(np.mean(seasonal_values))

    seasonals = [seasonal for i in range(length)]
    return pd.DataFrame([i for i in chain(*seasonals)][:length])


def get_models(models_file: str, model: Type, data: Iterable, params: List[tuple]) -> dict:
    """
    Строит модели по набору параметров. Обучение нескольких моделей - дело доолгое,
    поэтму сначла функция пытается загрузить модели из файла.
    Если модели строятся заново, то они сохраняются в указанный файл.

    :param models_file: имя файла с сериализованными моделями
    :param model: класс модели (ARMA)
    :param data: данные, на которых обучается модель
    :param params: параметры моделей (в случае ARMA - список разных порядков)
    :return: словарь с порядками моделей и самими моделями
    """
    try:
        if exists(models_file):
            print(f'File {models_file} found. Loading...')

            # Загрузим модели из файла.
            with open(models_file, 'rb') as serialized:
                models = pickle.load(serialized)
                print('Load success.')

                # Проверим, равны ли переданные параметры и параметры загруженных моделей
                if set(params) != set(models.keys()):
                    raise ValueError

                # Если всё в порядке, то вернём модельки
                return models
    except ValueError:
        print(f"Models in {models_file} don't match passed parameters or an error occurred.")

    print(f'Building models...')

    models = {param: model(data, param).fit() for param in params}

    print(f'Models built successfully')
    with open(models_file, 'wb') as serialized:
        pickle.dump(models, serialized)
        print(f'Models saved to {models_file}')

    return models


def get_best_model(models):
    """
    Получение порядка лучшей модели по критерию AIC или BIC.

    :param models: параметры моделей и их оценки
    :return: параметры лучшей модели
    """
    return [(int(x[0]), int(x[1])) for x in sorted(models, key=lambda x: x[2])][0]


def _make_features_for_trend(y_train: Sized, forecast_for: int) -> tuple:
    """
    Создает набор ворастающих целых чисел [0, len(y_train) + forecast_for],
    разбивая их на трейн и тест.

    :param y_train: набор данных для обучения
    :param forecast_for: количество значений, которые надо спрогнозированть
    :return: два набора фичей: [0, len(y_train)] и [len(y_train) + 1, len(y_train) + forecast_for]
    """
    full_range = np.array([np.array([i]) for i in range(len(y_train) + forecast_for)])
    return full_range[:len(y_train)], full_range[-forecast_for:]


def extrapolate_trend(
        y_trend_train: pd.DataFrame, forecast_for: int,
        model_type: Union[str, Type] = LinearRegression, degree: int = 1, *args, **kwargs) -> np.ndarray:
    """
    Экстраполирует трендовую составляющую на заданное количество значений вперед.
    Использует регрессионные модели или непараметрические.

    :param y_trend_train: известные значения трендовой составляющей
    :param forecast_for: количество значений, которые нужно спрогнозировать
    :param model_type: тип модели тренда. Если передаётся объект sklearn, то модель
                       обучается на трейне, и на основе неё прогнозируются значения
    :param degree: степень полинома для линейной модели тренда (игнорируется, если модель непараметрическая)
    :param args: произвольные аргументы, передаваемые в модель
    :param kwargs: произвольные аргументы, передаваемые в модель
    :return: массив numpy со значениями экстраполированного тренда
    """

    # Если модель тренда непараметрическая, то возвращаем пследнее известное значение
    if model_type in {'MA', 'ES', 'Nonparametric', 'nonparametric'}:
        return np.array([y_trend_train.values[-1] for _ in range(forecast_for)])

    # создадим фичи - массив возрастающих чисел [0, trend_size + test_size]
    X_trend_train, X_trend_test = _make_features_for_trend(y_trend_train, forecast_for)

    # Добавим полиномиальные фичи
    feature_maker = PolynomialFeatures(degree=degree)
    X_train = feature_maker.fit_transform(X_trend_train)
    y_train = y_trend_train

    # Обучим модель и предскажем нужные значения
    model = model_type(*args, **kwargs)
    model.fit(X_train, y_train)

    X_test = feature_maker.transform(X_trend_test)
    return model.predict(X_test)


def _detect_period(seasonal_data: pd.DataFrame) -> int:
    """
    Рассчитывает период по стабильному сезонному фильтру.

    :param seasonal_data: сезонная компонента ряда
    :return: период
    """
    initial = seasonal_data.values[0]
    index = 1
    while initial != seasonal_data.values[index]:
        index += 1
    return index


def extrapolate_seasonal(X_train: pd.DataFrame, forecast_for: int) -> Iterable:
    """
    Экстраполирует сезонную составляющую на forecast_for шагов вперед.
    Предполагается использование стабильного сезонного фильтра.

    :param X_train: известные значения сезонной компоненты временного ряда
    :param forecast_for: количество значений для экстраполяции
    :return: экстраполированные значения сезонной составляющей
    """
    length = _detect_period(X_train)
    period = X_train.values[:length]
    last_in_train = X_train.values[-1]

    surrounding = np.vstack([period for _ in range(forecast_for // length + 2)])

    index = 0
    while last_in_train != surrounding[index]:
        index += 1

    return surrounding[index + 1:][:forecast_for]
