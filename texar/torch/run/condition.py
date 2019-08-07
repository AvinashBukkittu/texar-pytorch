import functools
import types
from abc import ABC, abstractmethod
from enum import Enum, auto
from time import time as time_now
from typing import Any, Dict, Optional, Tuple

from texar.torch.run.executor_utils import MetricList
from texar.torch.utils.types import MaybeTuple

# pylint: disable=unused-argument

__all__ = [
    "Event",
    "EventPoint",
    "Condition",
    "epoch",
    "iteration",
    "validation",
    "consecutive",
    "time",
]


class Event(Enum):
    Iteration = auto()
    Epoch = auto()
    Training = auto()
    Validation = auto()
    ValidationIteration = auto()
    Testing = auto()
    TestingIteration = auto()
    ParameterUpdate = auto()


EventPoint = Tuple[Event, bool]


class Condition(ABC):
    _hooks: Dict[EventPoint, Any]

    @property
    @abstractmethod
    def _hash_attributes(self) -> MaybeTuple[Any]:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Condition):
            raise ValueError("Cannot compare to object not of type Condition")
        return self._hash_attributes == other._hash_attributes  # pylint: disable=protected-access

    def __hash__(self):
        return hash(self._hash_attributes)

    @property
    def hooks(self) -> Dict[EventPoint, Any]:
        return self._hooks

    def __init__(self):
        self._hooks = {}
        for hook_name in dir(self):
            if not hook_name.startswith("check_"):
                continue
            name = hook_name
            if name.endswith("_begin"):
                name = name[6:-6]
                point = False
            elif name.endswith("_end"):
                name = name[6:-4]
                point = True
            else:
                raise ValueError(
                    "Final part of hook name must be 'begin' or 'end'")
            if name not in Event.__members__:
                name = ''.join(x.capitalize() for x in name.split("_"))
                if name not in Event.__members__:
                    raise ValueError(
                        f"Hook name '{hook_name}' is not a valid event")
            event = Event.__members__[name]
            self._hooks[(event, point)] = getattr(self, hook_name)


class epoch(Condition):
    r"""Triggers when the specified number of epochs has ended.

    Args:
        num_epochs (int): The number of epochs to wait before triggering the
            event. In other words, the event is triggered every
            :attr:`num_epochs` epochs.
    """

    def __init__(self, num_epochs: int = 1):
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("`num_epochs` must be a positive integer")
        super().__init__()
        self.num_epochs = num_epochs
        self.count = 0

    @property
    def _hash_attributes(self):
        return self.num_epochs

    def check_epoch_end(self, executor) -> bool:
        self.count += 1
        if self.count == self.num_epochs:
            self.count = 0
            return True
        return False


class iteration(Condition):
    r"""Triggers when the specified number of iterations had ended.

    Args:
        num_iters (int): The number of iterations to wait before triggering the
            event. In other words, the event is triggered every
            :attr:`num_iters` iterations.
        mode (str): The mode under which iterations are counted. Available
            choices are ``"train"``, ``"valid"``, and ``"test"``. Defaults to
            ``"train"``.
    """

    def __new__(cls, num_iters: int = 1, mode: str = "train"):
        obj = super().__new__(cls)
        if mode == "train":
            obj.check_iteration_end = obj._check_iteration_end
        elif mode == "valid":
            obj.check_validation_iteration_end = obj._check_iteration_end
        elif mode == "test":
            obj.check_testing_iteration_end = obj._check_iteration_end
        else:
            raise ValueError(f"Invalid mode {mode}")
        return obj

    def __init__(self, num_iters: int = 1, mode: str = "train"):
        if not isinstance(num_iters, int) or num_iters <= 0:
            raise ValueError("`num_iters` must be a positive integer")
        super().__init__()
        self.num_iters = num_iters
        self.count = 0

    @property
    def _hash_attributes(self):
        return self.num_iters

    def _check_iteration_end(self, executor) -> bool:
        self.count += 1
        if self.count == self.num_iters:
            self.count = 0
            return True
        return False


class validation(Condition):
    r"""Triggers when validation ends, and optionally checks if validation
    results improve or worsen.

    Args:
        num_validations (int): The number of validations to wait before
            triggering the event. In other words, the event is triggered every
            :attr:`num_validations` validations.
        better (bool, optional): If `True`, this event only triggers when
            validation results improve; if `False`, only triggers when results
            worsen. Defaults to `None`, in which case the event triggers
            regardless of results.
    """

    def __init__(self, num_validations: int = 1, better: Optional[bool] = None):
        if not isinstance(num_validations, int) or num_validations <= 0:
            raise ValueError("`num_validations` must be a positive integer")
        super().__init__()
        self.num_valids = num_validations
        self.count = 0
        self.better = better
        self.prev_result: Optional[MetricList] = None

    @property
    def _hash_attributes(self):
        return self.num_valids, self.better

    def check_validation_end(self, executor) -> bool:
        self.count += 1
        if self.count < self.num_valids:
            return False
        self.count = 0
        if self.better is None:
            return True
        metrics = executor.status["eval_metric"]
        cur_result = MetricList(metrics)
        if self.prev_result is not None:
            better = cur_result > self.prev_result
        else:
            better = True
        if better:
            self.prev_result = cur_result
        return better == self.better


class consecutive(Condition):
    r"""Triggers when the specified condition passes checks for several times
    consecutively.

    For example: :python:`consecutive(validation(better=False), times=3)` would
    trigger if validation results do not improve for 3 times in a row.

    .. warning::

        This method might not work as you would expect.
        # TODO: Think about whether this is a good design.
        # TODO: Think about why I thought this might not work as I expected.

    Args:
        cond: The base condition to check.
        times (int): The number of times the base condition should pass checks
            consecutively.
        clear_after_trigger (bool): Whether the counter should be cleared after
            the event is triggered. If :attr:`clear_after_trigger` is set to
            `False`, once this event is triggered, it will trigger every time
            :attr:`cond` is triggered, until :attr:`cond` fails to trigger at
            some point. Defaults to `True`.
    """

    def __init__(self, cond: Condition, times: int,
                 clear_after_trigger: bool = True):
        super().__init__()
        self.cond = cond
        self.times = times
        self.count = 0
        self.clear_after_trigger = clear_after_trigger

        for hook_point, method in self.cond.hooks.items():
            self._hooks[hook_point] = self._create_check_method(method)

    @property
    def _hash_attributes(self):
        return self.cond, self.times, self.clear_after_trigger

    def _create_check_method(self, method):
        @functools.wraps(method)
        def check_fn(self, *args, **kwargs) -> bool:
            if method(*args, **kwargs):
                self.count += 1
                if self.count >= self.times:
                    if self.clear_after_trigger:
                        self.count = 0
                    return True
            else:
                self.count = 0
            return False

        return types.MethodType(check_fn, self)


class time(Condition):
    def __init__(self, *, hours: Optional[float] = None,
                 minutes: Optional[float] = None,
                 seconds: Optional[float] = None,
                 only_training: bool = True):
        super().__init__()
        self.seconds = 0.0
        if hours is not None:
            self.seconds += hours * 3600.0
        if minutes is not None:
            self.seconds += minutes * 60.0
        if seconds is not None:
            self.seconds += seconds
        self.only_training = only_training
        self.start_time: Optional[float] = None
        self.accumulated_time = 0.0

    @property
    def _hash_attributes(self):
        return self.seconds, self.only_training

    def _should_trigger(self) -> bool:
        total_time = self.accumulated_time
        if self.start_time is None:
            cur_time = None
        else:
            cur_time = time_now()
            total_time += cur_time - self.start_time
        self.start_time = cur_time
        if total_time >= self.seconds:
            self.accumulated_time = 0.0
            return True
        else:
            self.accumulated_time = total_time
            return False

    def check_training_begin(self, executor) -> bool:
        self.start_time = time_now()
        return False

    def check_training_end(self, executor) -> bool:
        return self._should_trigger()

    def check_validation_begin(self, executor) -> bool:
        if self.only_training and self.start_time is not None:
            self.accumulated_time += time_now() - self.start_time
            self.start_time = None
        return self._should_trigger()

    def check_validation_end(self, executor) -> bool:
        if self.only_training:
            self.start_time = time_now()
            return False
        else:
            return self._should_trigger()

    def check_testing_begin(self, executor) -> bool:
        if self.only_training and self.start_time is not None:
            self.accumulated_time += time_now() - self.start_time
            self.start_time = None
        return self._should_trigger()

    def check_testing_end(self, executor) -> bool:
        if self.only_training:
            self.start_time = time_now()
            return False
        else:
            return self._should_trigger()

    def check_iteration_end(self, executor) -> bool:
        return self._should_trigger()
