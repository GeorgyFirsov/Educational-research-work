import warnings
import functools


def deprecated(replacement: str):
    """
    Помечает функцию как устаревшую и выводит соответствующее
    предупреждение с названием функции, которую рекомендуется
    использовать вместо данной.

    @param replacement - имя новой функции
    """
    def deprecation_decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)

            warnings.warn(f'Function {func.__name__} is deprecated. Use {replacement} instead.',
                          category=DeprecationWarning,
                          stacklevel=2)

            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return wrapped
    return deprecation_decorator
