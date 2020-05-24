from typing import (Optional,
                    Iterable,
                    Union,
                    Dict,
                    )

import pandas as pd
from scipy.stats import jarque_bera
from scipy.stats import chi2
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima_model import ARMAResults, ARMAResultsWrapper
from statsmodels.stats.diagnostic import acorr_ljungbox


def print_test_results(test_name: str, results: Dict[str, float]) -> None:
    """
    Печатает результаты статистического теста для нескольких временных рядов
    в красивом формате.

    :param test_name: название статистического теста
    :param results: словарь из названия ряда и результата теста (p-value) на нем
    """

    print(f'Тест {test_name}. Значения p-value для временных рядов:')
    for series, pvalue in results.items():
        print(series, pvalue)


def dickey_fuller_test(
        time_series: Iterable, maxlag: Optional[int] = None,
        verbose: bool = False, return_pvalue: bool = False) -> Optional[float]:
    """
    Статистический тест Дики-Фуллера на стационарность.

    :param time_series: временной ряд
    :param maxlag: максимальное количество лагов, учитываемых тестом
    :param verbose: если True, то напечатается значение статистики критерия и p-value
    :param return_pvalue: если True, то тест вернет значение p-value, иначе - ничего
    :return: значение p-value или ничего
    """
    if not isinstance(time_series, pd.Series):
        time_series = pd.Series(time_series).dropna()

    # Теперь сам тест Дики-Фуллера
    test_result = adfuller(time_series, maxlag=maxlag, autolag='AIC' if maxlag is None else None)

    tstat = test_result[0:2][0]
    pvalue = test_result[0:2][1]

    if not return_pvalue:
        result_str = f'ряд {"" if pvalue < 0.05 else "не"}стационарный'
        print(f'Тест Дики-Фуллера: {result_str}')
        if verbose:
            print(f'Статистика критерия:\t{tstat}\np-value:\t\t{pvalue}')

    if return_pvalue:
        return pvalue


def ljung_box_test(
        time_series: Iterable, lags: Union[int, Iterable] = 1,
        verbose: bool = False, return_pvalue: bool = False) -> Optional[float]:
    """
    Q-тест Льюнга-Бокса на то, что ряд является белым шумом

    :param time_series: временной ряд
    :param lags: лаги, учитываемые тестом
    :param verbose: если True, то напечатается значение статистики критерия и p-value
    :param return_pvalue: если True, то тест вернет значение p-value, иначе - ничего
    :return: значение p-value или ничего
    """
    if not isinstance(time_series, pd.Series):
        time_series = pd.Series(time_series).dropna()

    # Теперь тест Льюнга-Бокса
    test_result = acorr_ljungbox(time_series, lags=lags)

    tstat = test_result[0][-1] if isinstance(lags, int) else test_result[0]
    pvalue = test_result[1][-1] if isinstance(lags, int) else test_result[1]

    if not return_pvalue:
        result_str = f'ряд{"" if pvalue > 0.05 else " не"} является слабым белым шумом'
        print(f'Тест Льюнга-Бокса: {result_str}')
        if verbose:
            print(f'Статистика критерия:\t{tstat}\np-value:\t\t{pvalue}')

    if return_pvalue:
        return pvalue


def kpss_test(
        time_series: Iterable, lags: Optional[Union[str, int]] = None,
        verbose: bool = False, return_pvalue: bool = False) -> Optional[float]:
    """
    KPSS тест на стационарность временного ряда

    :param time_series: временной ряд
    :param lags: лаги, учитываемые тестом
    :param verbose: если True, то напечатается значение статистики критерия и p-value
    :param return_pvalue: если True, то тест вернет значение p-value, иначе - ничего
    :return: значение p-value или ничего
    """
    if not isinstance(time_series, pd.Series):
        time_series = pd.Series(time_series).dropna()

    # Теперь тест КПСС
    test_result = kpss(time_series, lags=lags)

    tstat = test_result[0]
    pvalue = test_result[1]

    if not return_pvalue:
        result_str = f'ряд {"" if pvalue > 0.05 else "не"}стационарный'
        print(f'Тест KPSS: {result_str}')
        if verbose:
            print(f'Статистика критерия:\t{tstat}\np-value:\t\t{pvalue}')

    if return_pvalue:
        return pvalue


def jarque_bera_test(
        time_series: Iterable,
        verbose: bool = False, return_pvalue: bool = False) -> Optional[float]:
    """
    Статистический тест Харке-Бера на нормальность распределения значений
    во временном ряде

    :param time_series: временной ряд
    :param verbose: если True, то напечатается значение статистики критерия и p-value
    :param return_pvalue: если True, то тест вернет значение p-value, иначе - ничего
    :return: значение p-value или ничего
    """
    if not isinstance(time_series, pd.Series):
        time_series = pd.Series(time_series).dropna()

    # Теперь сам тест Харке-Бера
    test_result = jarque_bera(time_series)

    tstat = test_result[0]
    pvalue = test_result[1]

    if not return_pvalue:
        result_str = f'значения{"" if pvalue > 0.05 else " не"} имеют нормальное распределение'
        print(f'Тест Харке-Бера: {result_str}')
        if verbose:
            print(f'Статистика критерия:\t{tstat}\np-value:\t\t{pvalue}')

    if return_pvalue:
        return pvalue


def likelyhood_ratio_test(
        l1: Union[float, ARMAResults], l2: Union[float, ARMAResults],
        verbose: bool = False, return_pvalue: bool = False) -> Optional[float]:
    """
    Статистический LR тест на статистическую значемость различия оценок
    соответствия моделей данным.

    :param l1: результат обучения первой ARMA модели или ее логарифм функции правдоподобия
    :param l2: результат обучения второй ARMA модели или ее логарифм функции правдоподобия
    :param verbose: если True, то напечатается значение статистики критерия и p-value
    :param return_pvalue: если True, то тест вернет значение p-value, иначе - ничего
    :return: значение p-value или ничего
    """
    if isinstance(l1, ARMAResults) or isinstance(l1, ARMAResultsWrapper):
        l1 = l1.llf
    if isinstance(l2, ARMAResults) or isinstance(l2, ARMAResultsWrapper):
        l2 = l2.llf

    tstat = 2 * (l1 - l2)
    pvalue = chi2.sf(tstat, 1)

    if not return_pvalue:
        result_str = f'значения{"" if pvalue < 0.05 else " не"} отличаются значительно'
        print(f'LR тест: {result_str}')
        if verbose:
            print(f'Статистика критерия:\t{tstat}\np-value:\t\t{pvalue}')

    if return_pvalue:
        return pvalue
