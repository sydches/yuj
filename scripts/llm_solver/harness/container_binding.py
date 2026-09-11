"""Keep discovered container images consistent within one task invocation."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
import os
from threading import RLock

from .sandbox.container_backend import ContainerBackendError
from .time_budget import execution_deadline, remaining_before


def _connection_selectors():
    # Detect changed declarations, without claiming to pin the responding engine,
    # TLS material or remote connection.
    return tuple((name, os.environ.get(name)) for name in (
        'DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_CONFIG',
        'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH',
        'CONTAINER_HOST', 'CONTAINER_CONNECTION', 'CONTAINERS_CONF',
    ))


class ContainerImageBinding:
    """Share one discovery across startup, file readers and command consumers."""

    def __init__(self):
        self._lock = RLock()
        self._request = None
        self._backend = None

    def bind(self, backend, runtime_bin, *, timeout=None):
        selectors = _connection_selectors()
        request = (backend, runtime_bin, selectors)
        with self._lock:
            remaining_before(execution_deadline(), timeout)
            if self._request is not None:
                if request not in (self._request, (self._backend, *self._request[1:])):
                    raise ContainerBackendError(
                        'container execution selection changed after image binding')
                return self._backend
            digest = backend.image_digest(runtime_bin, timeout=timeout)
            if selectors != _connection_selectors():
                raise ContainerBackendError(
                    'container execution selection changed during image discovery')
            self._backend = replace(backend, image=digest)
            self._request = request
            return self._backend


_ACTIVE = ContextVar('container_image_binding', default=None)


def current_container_image_binding():
    return _ACTIVE.get()


def has_bound_container_image():
    binding = _ACTIVE.get()
    return binding is not None and binding._backend is not None


@contextmanager
def container_image_scope(binding=None, *, fresh=False):
    """Reuse the invocation's discovery, or own an isolated standalone scope."""
    active = _ACTIVE.get()
    if binding is not None and active is not None and binding is not active:
        raise ContainerBackendError('cannot replace an active container image binding')
    selected = binding if binding is not None else (
        ContainerImageBinding() if fresh or active is None else active)
    token = _ACTIVE.set(selected)
    try:
        yield selected
    finally:
        _ACTIVE.reset(token)


def bind_container_image(backend, runtime_bin, *, timeout=None):
    binding = _ACTIVE.get()
    if binding is None:
        binding = ContainerImageBinding()
    return binding.bind(backend, runtime_bin, timeout=timeout)


def container_scoped_session(function):
    """Retain discovery between direct Session construction and execution."""
    @wraps(function)
    def scoped(session, *args, **kwargs):
        with container_image_scope(getattr(session, '_container_image_binding', None)) as binding:
            session._container_image_binding = binding
            return function(session, *args, **kwargs)
    return scoped
