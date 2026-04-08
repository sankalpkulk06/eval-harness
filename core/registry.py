from typing import Type

from core.base import BaseGenerator, BaseMetric, BaseRetriever

RETRIEVER_REGISTRY: dict[str, Type[BaseRetriever]] = {}
GENERATOR_REGISTRY: dict[str, Type[BaseGenerator]] = {}
METRIC_REGISTRY: dict[str, Type[BaseMetric]] = {}


def register_retriever(name: str):
    """Decorator to register a retriever class."""

    def decorator(cls: Type[BaseRetriever]) -> Type[BaseRetriever]:
        RETRIEVER_REGISTRY[name] = cls
        return cls

    return decorator


def register_generator(name: str):
    """Decorator to register a generator class."""

    def decorator(cls: Type[BaseGenerator]) -> Type[BaseGenerator]:
        GENERATOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_metric(name: str):
    """Decorator to register a metric class."""

    def decorator(cls: Type[BaseMetric]) -> Type[BaseMetric]:
        METRIC_REGISTRY[name] = cls
        return cls

    return decorator
