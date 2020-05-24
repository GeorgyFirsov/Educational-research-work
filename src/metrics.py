from typing import (Sized,
                    Iterable,
                    Union,
                    )

import pandas as pd
import numpy as np


def _ensure_length_equality(sample1: Sized, sample2: Sized) -> None:
    """
    Проверяет равенство длин данных и кидает исключение в случае неравенства

    :raises ValueError: при неравенстве длин
    :param sample1: первая выборка значений
    :param sample2: вторая выборка значений
    """

    if len(sample1) != len(sample2):
        raise ValueError(
            f'''Data samples must have the same length. 
            Actual lengths: {len(sample1)}, {len(sample2)}''')


def _convert_from_df_if_necessary(sample: Union[Iterable, Sized, pd.DataFrame]) -> Union[Iterable, Sized]:
    """
    Если на входе датафрейм, то вернется низлежащий массив numpy

    :param sample: набор данных (может быть датафреймом pandas)
    :return: массив numpy или иной iterable (список, кортеж)
    """
    return sample.values if isinstance(sample, pd.DataFrame) else sample


def mean_absolute_percentage_error(
        original: Union[Iterable, Sized],
        forecasted: Union[Iterable, Sized]) -> float:
    """
    Рассчитывает среднюю абсолютную процентную ошибку. Формула:
            MAPE = sum(|(o - f) / o|) / n

    :param original: истинные (эталонные) значения
    :param forecasted: спрогнозированные значения
    :return: значение MAPE
    """

    # Проверка равенства длин
    _ensure_length_equality(original, forecasted)

    # С датафреймами работать сложнее
    original = _convert_from_df_if_necessary(original)
    forecasted = _convert_from_df_if_necessary(forecasted)

    n = len(original)
    result = sum([abs((o - f) / o) for o, f in zip(original, forecasted)]) / n
    return result[0] if isinstance(result, np.ndarray) else result


def mean_squared_error(
        original: Union[Iterable, Sized],
        forecasted: Union[Iterable, Sized]) -> float:
    """
    Рассчитывает среднюю квадратичную ошибку. Формула:
            MSE = sum((o - f) ^ 2) / n

    :param original: истинные (эталонные) значения
    :param forecasted: спрогнозированные значения
    :return: значение MSE
    """

    # Проверка равенства длин
    _ensure_length_equality(original, forecasted)

    # С датафреймами работать сложнее
    original = _convert_from_df_if_necessary(original)
    forecasted = _convert_from_df_if_necessary(forecasted)

    n = len(original)
    result = sum([(o - f) ** 2 for o, f in zip(original, forecasted)]) / n
    return result[0] if isinstance(result, np.ndarray) else result
