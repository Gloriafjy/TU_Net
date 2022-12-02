import functools
import logging
from contextlib import contextmanager
import inspect
import time

logger = logging.getLogger(__name__)


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


def deserialize_model(package, strict=False):
    klass = package['class']
    if strict:
        model = klass(*package['args'], **package['kwargs'])
    else:
        sig = inspect.signature(klass)
        kw = package['kwargs']
        for key in list(kw):
            if key not in sig.parameters:
                logger.warning("Dropping inexistant parameter %s", key)
                del kw[key]
        model = klass(*package['args'], **kw)
    model.load_state_dict(package['state'])
    return model


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def serialize_model(model):
    args, kwargs = model._init_args_kwargs
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "args": args, "kwargs": kwargs, "state": state}


@contextmanager
def swap_state(model, state):
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out


class LogProgress:
    def __init__(self,
                 logger,
                 iterable,
                 updates=5,
                 total=None,
                 name="LogProgress",
                 level=logging.INFO):
        self.iterable = iterable
        self.total = total or len(iterable)
        self.updates = updates
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            log_every = max(1, self.total // self.updates)
            if self._index >= 1 and self._index % log_every == 0:
                self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k.capitalize()} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1/self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.1f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)


def colorize(text, color):
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    return colorize(text, "1")
