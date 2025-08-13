"""threadpoolctl

This module provides utilities to introspect native libraries that relies on
thread pools (notably BLAS and OpenMP implementations) and dynamically set the
maximal number of threads they can use.
"""

# License: BSD 3-Clause

# The code to introspect dynamically loaded libraries on POSIX systems is
# adapted from code by Intel developer @anton-malakhov available at
# https://github.com/IntelPython/smp (Copyright (c) 2017, Intel Corporation)
# and also published under the BSD 3-Clause license
import os
import re
import sys
import ctypes
import itertools
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator

__version__ = "3.6.0"
__all__ = [
    "threadpool_limits",
    "threadpool_info",
    "ThreadpoolController",
    "LibController",
    "register",
]


# One can get runtime errors or even segfaults due to multiple OpenMP libraries
# loaded simultaneously which can happen easily in Python when importing and
# using compiled extensions built with different compilers and therefore
# different OpenMP runtimes in the same program. In particular libiomp (used by
# Intel ICC) and libomp used by clang/llvm tend to crash. This can happen for
# instance when calling BLAS inside a prange. Setting the following environment
# variable allows multiple OpenMP libraries to be loaded. It should not degrade
# performances since we manually take care of potential over-subscription
# performance issues, in sections of the code where nested OpenMP loops can
# happen, by dynamically reconfiguring the inner OpenMP runtime to temporarily
# disable it while under the scope of the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Structure to cast the info on dynamically loaded library. See
# https://linux.die.net/man/3/dl_iterate_phdr for more details.
_SYSTEM_UINT = ctypes.c_uint64 if sys.maxsize > 2**32 else ctypes.c_uint32
_SYSTEM_UINT_HALF = ctypes.c_uint32 if sys.maxsize > 2**32 else ctypes.c_uint16


class _dl_phdr_info(ctypes.Structure):
    _fields_ = [
        ("dlpi_addr", _SYSTEM_UINT),  # Base address of object
        ("dlpi_name", ctypes.c_char_p),  # path to the library
        ("dlpi_phdr", ctypes.c_void_p),  # pointer on dlpi_headers
        ("dlpi_phnum", _SYSTEM_UINT_HALF),  # number of elements in dlpi_phdr
    ]


# The RTLD_NOLOAD flag for loading shared libraries is not defined on Windows.
try:
    _RTLD_NOLOAD = os.RTLD_NOLOAD
except AttributeError:
    _RTLD_NOLOAD = ctypes.DEFAULT_MODE


class LibController(ABC):
    """Abstract base class for the individual library controllers

    A library controller must expose the following class attributes:
        - user_api : str
            Usually the name of the library or generic specification the library
            implements, e.g. "blas" is a specification with different implementations.
        - internal_api : str
            Usually the name of the library or concrete implementation of some
            specification, e.g. "openblas" is an implementation of the "blas"
            specification.
        - filename_prefixes : tuple
            Possible prefixes of the shared library's filename that allow to
            identify the library. e.g. "libopenblas" for libopenblas.so.

    and implement the following methods: `get_num_threads`, `set_num_threads` and
    `get_version`.

    Threadpoolctl loops through all the loaded shared libraries and tries to match
    the filename of each library with the `filename_prefixes`. If a match is found, a
    controller is instantiated and a handler to the library is stored in the `dynlib`
    attribute as a `ctypes.CDLL` object. It can be used to access the necessary symbols
    of the shared library to implement the above methods.

    The following information will be exposed in the info dictionary:
      - user_api : standardized API, if any, or a copy of internal_api.
      - internal_api : implementation-specific API.
      - num_threads : the current thread limit.
      - prefix : prefix of the shared library's filename.
      - filepath : path to the loaded shared library.
      - version : version of the library (if available).

    In addition, each library controller may expose internal API specific entries. They
    must be set as attributes in the `set_additional_attributes` method.
    """

    @final
    def __init__(self, *, filepath=None, prefix=None, parent=None):
        """This is not meant to be overriden by subclasses."""
        self.parent = parent
        self.prefix = prefix
        self.filepath = filepath
        self.dynlib = ctypes.CDLL(filepath, mode=_RTLD_NOLOAD)
        self._symbol_prefix, self._symbol_suffix = self._find_affixes()
        self.version = self.get_version()
        self.set_additional_attributes()

    def info(self):
        """Return relevant info wrapped in a dict"""
        hidden_attrs = ("dynlib", "parent", "_symbol_prefix", "_symbol_suffix")
        return {
            "user_api": self.user_api,
            "internal_api": self.internal_api,
            "num_threads": self.num_threads,
            **{k: v for k, v in vars(self).items() if k not in hidden_attrs},
        }

    def set_additional_attributes(self):
        """Set additional attributes meant to be exposed in the info dict"""

    @property
    def num_threads(self):
        """Exposes the current thread limit as a dynamic property

        This is not meant to be used or overriden by subclasses.
        """
        return self.get_num_threads()

    @abstractmethod
    def get_num_threads(self):
        """Return the maximum number of threads available to use"""

    @abstractmethod
    def set_num_threads(self, num_threads):
        """Set the maximum number of threads to use"""

    @abstractmethod
    def get_version(self):
        """Return the version of the shared library"""

    def _find_affixes(self):
        """Return the affixes for the symbols of the shared library"""
        return "", ""

    def _get_symbol(self, name):
        """Return the symbol of the shared library accounding for the affixes"""
        return getattr(
            self.dynlib, f"{self._symbol_prefix}{name}{self._symbol_suffix}", None
        )


class OpenBLASController(LibController):
    """Controller class for OpenBLAS"""

    user_api = "blas"
    internal_api = "openblas"
    filename_prefixes = ("libopenblas", "libblas", "libscipy_openblas")

    _symbol_prefixes = ("", "scipy_")
    _symbol_suffixes = ("", "64_", "_64")

    # All variations of "openblas_get_num_threads", accounting for the affixes
    check_symbols = tuple(
        f"{prefix}openblas_get_num_threads{suffix}"
        for prefix, suffix in itertools.product(_symbol_prefixes, _symbol_suffixes)
    )

    def _find_affixes(self):
        for prefix, suffix in itertools.product(
            self._symbol_prefixes, self._symbol_suffixes
        ):
            if hasattr(self.dynlib, f"{prefix}openblas_get_num_threads{suffix}"):
                return prefix, suffix

    def set_additional_attributes(self):
        self.threading_layer = self._get_threading_layer()
        self.architecture = self._get_architecture()

    def get_num_threads(self):
        get_num_threads_func = self._get_symbol("openblas_get_num_threads")
        if get_num_threads_func is not None:
            return get_num_threads_func()
        return None

    def set_num_threads(self, num_threads):
        set_num_threads_func = self._get_symbol("openblas_set_num_threads")
        if set_num_threads_func is not None:
            return set_num_threads_func(num_threads)
        return None

    def get_version(self):
        # None means OpenBLAS is not loaded or version < 0.3.4, since OpenBLAS
        # did not expose its version before that.
        get_version_func = self._get_symbol("openblas_get_config")
        if get_version_func is not None:
            get_version_func.restype = ctypes.c_char_p
            config = get_version_func().split()
            if config[0] == b"OpenBLAS":
                return config[1].decode("utf-8")
            return None
        return None

    def _get_threading_layer(self):
        """Return the threading layer of OpenBLAS"""
        get_threading_layer_func = self._get_symbol("openblas_get_parallel")
        if get_threading_layer_func is not None:
            threading_layer = get_threading_layer_func()
            if threading_layer == 2:
                return "openmp"
            elif threading_layer == 1:
                return "pthreads"
            return "disabled"
        return "unknown"

    def _get_architecture(self):
        """Return the architecture detected by OpenBLAS"""
        get_architecture_func = self._get_symbol("openblas_get_corename")
        if get_architecture_func is not None:
            get_architecture_func.restype = ctypes.c_char_p
            return get_architecture_func().decode("utf-8")
        return None


class BLISController(LibController):
    """Controller class for BLIS"""

    user_api = "blas"
    internal_api = "blis"
    filename_prefixes = ("libblis", "libblas")
    check_symbols = (
        "bli_thread_get_num_threads",
        "bli_thread_set_num_threads",
        "bli_info_get_version_str",
        "bli_info_get_enable_openmp",
        "bli_info_get_enable_pthreads",
        "bli_arch_query_id",
        "bli_arch_string",
    )

    def set_additional_attributes(self):
        self.threading_layer = self._get_threading_layer()
        self.architecture = self._get_architecture()

    def get_num_threads(self):
        get_func = getattr(self.dynlib, "bli_thread_get_num_threads", lambda: None)
        num_threads = get_func()
        # by default BLIS is single-threaded and get_num_threads
        # returns -1. We map it to 1 for consistency with other libraries.
        return 1 if num_threads == -1 else num_threads

    def set_num_threads(self, num_threads):
        set_func = getattr(
            self.dynlib, "bli_thread_set_num_threads", lambda num_threads: None
        )
        return set_func(num_threads)

    def get_version(self):
        get_version_ = getattr(self.dynlib, "bli_info_get_version_str", None)
        if get_version_ is None:
            return None

        get_version_.restype = ctypes.c_char_p
        return get_version_().decode("utf-8")

    def _get_threading_layer(self):
        """Return the threading layer of BLIS"""
        if getattr(self.dynlib, "bli_info_get_enable_openmp", lambda: False)():
            return "openmp"
        elif getattr(self.dynlib, "bli_info_get_enable_pthreads", lambda: False)():
            return "pthreads"
        return "disabled"

    def _get_architecture(self):
        """Return the architecture detected by BLIS"""
        bli_arch_query_id = getattr(self.dynlib, "bli_arch_query_id", None)
        bli_arch_string = getattr(self.dynlib, "bli_arch_string", None)
        if bli_arch_query_id is None or bli_arch_string is None:
            return None

        # the true restype should be BLIS' arch_t (enum) but int should work
        # for us:
        bli_arch_query_id.restype = ctypes.c_int
        bli_arch_string.restype = ctypes.c_char_p
        return bli_arch_string(bli_arch_query_id()).decode("utf-8")


class FlexiBLASController(LibController):
    """Controller class for FlexiBLAS"""

    user_api = "blas"
    internal_api = "flexiblas"
    filename_prefixes = ("libflexiblas",)
    check_symbols = (
        "flexiblas_get_num_threads",
        "flexiblas_set_num_threads",
        "flexiblas_get_version",
        "flexiblas_list",
        "flexiblas_list_loaded",
        "flexiblas_current_backend",
    )

    @property
    def loaded_backends(self):
        return self._get_backend_list(loaded=True)

    @property
    def current_backend(self):
        return self._get_current_backend()

    def info(self):
        """Return relevant info wrapped in a dict"""
        # We override the info method because the loaded and current backends
        # are dynamic properties
        exposed_attrs = super().info()
        exposed_attrs["loaded_backends"] = self.loaded_backends
        exposed_attrs["current_backend"] = self.current_backend

        return exposed_attrs

    def set_additional_attributes(self):
        self.available_backends = self._get_backend_list(loaded=False)

    def get_num_threads(self):
        get_func = getattr(self.dynlib, "flexiblas_get_num_threads", lambda: None)
        num_threads = get_func()
        # by default BLIS is single-threaded and get_num_threads
        # returns -1. We map it to 1 for consistency with other libraries.
        return 1 if num_threads == -1 else num_threads

    def set_num_threads(self, num_threads):
        set_func = getattr(
            self.dynlib, "flexiblas_set_num_threads", lambda num_threads: None
        )
        return set_func(num_threads)

    def get_version(self):
        get_version_ = getattr(self.dynlib, "flexiblas_get_version", None)
        if get_version_ is None:
            return None

        major = ctypes.c_int()
        minor = ctypes.c_int()
        patch = ctypes.c_int()
        get_version_(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
        return f"{major.value}.{minor.value}.{patch.value}"

    def _get_backend_list(self, loaded=False):
        """Return the list of available backends for FlexiBLAS.

        If loaded is False, return the list of available backends from the FlexiBLAS
        configuration. If loaded is True, return the list of actually loaded backends.
        """
        func_name = f"flexiblas_list{'_loaded' if loaded else ''}"
        get_backend_list_ = getattr(self.dynlib, func_name, None)
        if get_backend_list_ is None:
            return None

        n_backends = get_backend_list_(None, 0, 0)

        backends = []
        for i in range(n_backends):
            backend_name = ctypes.create_string_buffer(1024)
            get_backend_list_(backend_name, 1024, i)
            if backend_name.value.decode("utf-8") != "__FALLBACK__":
                # We don't know when to expect __FALLBACK__ but it is not a real
                # backend and does not show up when running flexiblas list.
                backends.append(backend_name.value.decode("utf-8"))
        return backends

    def _get_current_backend(self):
        """Return the backend of FlexiBLAS"""
        get_backend_ = getattr(self.dynlib, "flexiblas_current_backend", None)
        if get_backend_ is None:
            return None

        backend = ctypes.create_string_buffer(1024)
        get_backend_(backend, ctypes.sizeof(backend))
        return backend.value.decode("utf-8")

    def switch_backend(self, backend):
        """Switch the backend of FlexiBLAS

        Parameters
        ----------
        backend : str
            The name or the path to the shared library of the backend to switch to. If
            the backend is not already loaded, it will be loaded first.
        """
        if backend not in self.loaded_backends:
            if backend in self.available_backends:
                load_func = getattr(self.dynlib, "flexiblas_load_backend", lambda _: -1)
            else:  # assume backend is a path to a shared library
                load_func = getattr(
                    self.dynlib, "flexiblas_load_backend_library", lambda _: -1
                )
            res = load_func(str(backend).encode("utf-8"))
            if res == -1:
                raise RuntimeError(
                    f"Failed to load backend {backend!r}. It must either be the name of"
                    " a backend available in the FlexiBLAS configuration "
                    f"{self.available_backends} or the path to a valid shared library."
                )

            # Trigger a new search of loaded shared libraries since loading a new
            # backend caused a dlopen.
            self.parent._load_libraries()

        switch_func = getattr(self.dynlib, "flexiblas_switch", lambda _: -1)
        idx = self.loaded_backends.index(backend)
        res = switch_func(idx)
        if res == -1:
            raise RuntimeError(f"Failed to switch to backend {backend!r}.")


class MKLController(LibController):
    """Controller class for MKL"""

    user_api = "blas"
    internal_api = "mkl"
    filename_prefixes = ("libmkl_rt", "mkl_rt", "libblas")
    check_symbols = (
        "MKL_Get_Max_Threads",
        "MKL_Set_Num_Threads",
        "MKL_Get_Version_String",
        "MKL_Set_Threading_Layer",
    )

    def set_additional_attributes(self):
        self.threading_layer = self._get_threading_layer()

    def get_num_threads(self):
        get_func = getattr(self.dynlib, "MKL_Get_Max_Threads", lambda: None)
        return get_func()

    def set_num_threads(self, num_threads):
        set_func = getattr(self.dynlib, "MKL_Set_Num_Threads", lambda num_threads: None)
        return set_func(num_threads)

    def get_version(self):
        if not hasattr(self.dynlib, "MKL_Get_Version_String"):
            return None

        res = ctypes.create_string_buffer(200)
        self.dynlib.MKL_Get_Version_String(res, 200)

        version = res.value.decode("utf-8")
        group = re.search(r"Version ([^ ]+) ", version)
        if group is not None:
            version = group.groups()[0]
        return version.strip()

    def _get_threading_layer(self):
        """Return the threading layer of MKL"""
        # The function mkl_set_threading_layer returns the current threading
        # layer. Calling it with an invalid threading layer allows us to safely
        # get the threading layer
        set_threading_layer = getattr(
            self.dynlib, "MKL_Set_Threading_Layer", lambda layer: -1
        )
        layer_map = {
            0: "intel",
            1: "sequential",
            2: "pgi",
            3: "gnu",
            4: "tbb",
            -1: "not specified",
        }
        return layer_map[set_threading_layer(-1)]


class OpenMPController(LibController):
    """Controller class for OpenMP"""

    user_api = "openmp"
    internal_api = "openmp"
    filename_prefixes = ("libiomp", "libgomp", "libomp", "vcomp")
    check_symbols = (
        "omp_get_max_threads",
        "omp_get_num_threads",
    )

    def get_num_threads(self):
        get_func = getattr(self.dynlib, "omp_get_max_threads", lambda: None)
        return get_func()

    def set_num_threads(self, num_threads):
        set_func = getattr(self.dynlib, "omp_set_num_threads", lambda num_threads: None)
        return set_func(num_threads)

    def get_version(self):
        # There is no way to get the version number programmatically in OpenMP.
        return None


# Controllers for the libraries that we'll look for in the loaded libraries.
# Third party libraries can register their own controllers.
_ALL_CONTROLLERS = [
    OpenBLASController,
    BLISController,
    MKLController,
    OpenMPController,
    FlexiBLASController,
]

# Helpers for the doc and test names
_ALL_USER_APIS = list(set(lib.user_api for lib in _ALL_CONTROLLERS))
_ALL_INTERNAL_APIS = [lib.internal_api for lib in _ALL_CONTROLLERS]
_ALL_PREFIXES = list(
    set(prefix for lib in _ALL_CONTROLLERS for prefix in lib.filename_prefixes)
)
_ALL_BLAS_LIBRARIES = [
    lib.internal_api for lib in _ALL_CONTROLLERS if lib.user_api == "blas"
]
_ALL_OPENMP_LIBRARIES = OpenMPController.filename_prefixes


def register(controller):
    """Register a new controller"""
    _ALL_CONTROLLERS.append(controller)
    _ALL_USER_APIS.append(controller.user_api)
    _ALL_INTERNAL_APIS.append(controller.internal_api)
    _ALL_PREFIXES.extend(controller.filename_prefixes)


def _format_docstring(*args, **kwargs):
    def decorator(o):
        if o.__doc__ is not None:
            o.__doc__ = o.__doc__.format(*args, **kwargs)
        return o

    return decorator


@lru_cache(maxsize=10000)
def _realpath(filepath):
    """Small caching wrapper around os.path.realpath to limit system calls"""
    return os.path.realpath(filepath)


@_format_docstring(USER_APIS=list(_ALL_USER_APIS), INTERNAL_APIS=_ALL_INTERNAL_APIS)
def threadpool_info():
    """Return the maximal number of threads for each detected library.

    Return a list with all the supported libraries that have been found. Each
    library is represented by a dict with the following information:

      - "user_api" : user API. Possible values are {USER_APIS}.
      - "internal_api": internal API. Possible values are {INTERNAL_APIS}.
      - "prefix" : filename prefix of the specific implementation.
      - "filepath": path to the loaded library.
      - "version": version of the library (if available).
      - "num_threads": the current thread limit.

    In addition, each library may contain internal_api specific entries.
    """
    return ThreadpoolController().info()


class _ThreadpoolLimiter:
    """The guts of ThreadpoolController.limit

    Refer to the docstring of ThreadpoolController.limit for more details.

    It will only act on the library controllers held by the provided `controller`.
    Using the default constructor sets the limits right away such that it can be used as
    a callable. Setting the limits can be delayed by using the `wrap` class method such
    that it can be used as a decorator.
    """

    def __init__(self, controller, *, limits=None, user_api=None):
        self._controller = controller
        self._limits, self._user_api, self._prefixes = self._check_params(
            limits, user_api
        )
        self._original_info = self._controller.info()
        self._set_threadpool_limits()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.restore_original_limits()

    @classmethod
    def wrap(cls, controller, *, limits=None, user_api=None):
        """Return an instance of this class that can be used as a decorator"""
        return _ThreadpoolLimiterDecorator(
            controller=controller, limits=limits, user_api=user_api
        )

    def restore_original_limits(self):
        """Set the limits back to their original values"""
        for lib_controller, original_info in zip(
            self._controller.lib_controllers, self._original_info
        ):
            lib_controller.set_num_threads(original_info["num_threads"])

    # Alias of `restore_original_limits` for backward compatibility
    unregister = restore_original_limits

    def get_original_num_threads(self):
        """Original num_threads from before calling threadpool_limits

        Return a dict `{user_api: num_threads}`.
        """
        num_threads = {}
        warning_apis = []

        for user_api in self._user_api:
            limits = [
                lib_info["num_threads"]
                for lib_info in self._original_info
                if lib_info["user_api"] == user_api
            ]
            limits = set(limits)
            n_limits = len(limits)

            if n_limits == 1:
                limit = limits.pop()
            elif n_limits == 0:
                limit = None
            else:
                limit = min(limits)
                warning_apis.append(user_api)

            num_threads[user_api] = limit

        if warning_apis:
            warnings.warn(
                "Multiple value possible for following user apis: "
                + ", ".join(warning_apis)
                + ". Returning the minimum."
            )

        return num_threads

    def _check_params(self, limits, user_api):
        """Suitable values for the _limits, _user_api and _prefixes attributes"""

        if isinstance(limits, str) and limits == "sequential_blas_under_openmp":
            (
                limits,
                user_api,
            ) = self._controller._get_params_for_sequential_blas_under_openmp().values()

        if limits is None or isinstance(limits, int):
            if user_api is None:
                user_api = _ALL_USER_APIS
            elif user_api in _ALL_USER_APIS:
                user_api = [user_api]
            else:
                raise ValueError(
                    f"user_api must be either in {_ALL_USER_APIS} or None. Got "
                    f"{user_api} instead."
                )

            if limits is not None:
                limits = {api: limits for api in user_api}
            prefixes = []
        else:
            if isinstance(limits, list):
                # This should be a list of dicts of library info, for
                # compatibility with the result from threadpool_info.
                limits = {
                    lib_info["prefix"]: lib_info["num_threads"] for lib_info in limits
                }
            elif isinstance(limits, ThreadpoolController):
                # To set the limits from the library controllers of a
                # ThreadpoolController object.
                limits = {
                    lib_controller.prefix: lib_controller.num_threads
                    for lib_controller in limits.lib_controllers
                }

            if not isinstance(limits, dict):
                raise TypeError(
                    "limits must either be an int, a list, a dict, or "
                    f"'sequential_blas_under_openmp'. Got {type(limits)} instead"
                )

            # With a dictionary, can set both specific limit for given
            # libraries and global limit for user_api. Fetch each separately.
            prefixes = [prefix for prefix in limits if prefix in _ALL_PREFIXES]
            user_api = [api for api in limits if api in _ALL_USER_APIS]

        return limits, user_api, prefixes

    def _set_threadpool_limits(self):
        """Change the maximal number of threads in selected thread pools.

        Return a list with all the supported libraries that have been found
        matching `self._prefixes` and `self._user_api`.
        """
        if self._limits is None:
            return

        for lib_controller in self._controller.lib_controllers:
            # self._limits is a dict {key: num_threads} where key is either
            # a prefix or a user_api. If a library matches both, the limit
            # corresponding to the prefix is chosen.
            if lib_controller.prefix in self._limits:
                num_threads = self._limits[lib_controller.prefix]
            elif lib_controller.user_api in self._limits:
                num_threads = self._limits[lib_controller.user_api]
            else:
                continue

            if num_threads is not None:
                lib_controller.set_num_threads(num_threads)


class _ThreadpoolLimiterDecorator(_ThreadpoolLimiter, ContextDecorator):
    """Same as _ThreadpoolLimiter but to be used as a decorator"""

    def __init__(self, controller, *, limits=None, user_api=None):
        self._limits, self._user_api, self._prefixes = self._check_params(
            limits, user_api
        )
        self._controller = controller

    def __enter__(self):
        # we need to set the limits here and not in the __init__ because we want the
        # limits to be set when calling the decorated function, not when creating the
        # decorator.
        self._original_info = self._controller.info()
        self._set_threadpool_limits()
        return self


@_format_docstring(
    USER_APIS=", ".join(f'"{api}"' for api in _ALL_USER_APIS),
    BLAS_LIBS=", ".join(_ALL_BLAS_LIBRARIES),
    OPENMP_LIBS=", ".join(_ALL_OPENMP_LIBRARIES),
)
class threadpool_limits(_ThreadpoolLimiter):
    """Change the maximal number of threads that can be used in thread pools.

    This object can be used either as a callable (the construction of this object
    limits the number of threads), as a context manager in a `with` block to
    automatically restore the original state of the controlled libraries when exiting
    the block, or as a decorator through its `wrap` method.

    Set the maximal number of threads that can be used in thread pools used in
    the supported libraries to `limit`. This function works for libraries that
    are already loaded in the interpreter and can be changed dynamically.

    This effect is global and impacts the whole Python process. There is no thread level
    isolation as these libraries do not offer thread-local APIs to configure the number
    of threads to use in nested parallel calls.

    Parameters
    ----------
    limits : int, dict, 'sequential_blas_under_openmp' or None (default=None)
        The maximal number of threads that can be used in thread pools

        - If int, sets the maximum number of threads to `limits` for each
          library selected by `user_api`.

        - If it is a dictionary `{{key: max_threads}}`, this function sets a
          custom maximum number of threads for each `key` which can be either a
          `user_api` or a `prefix` for a specific library.

        - If 'sequential_blas_under_openmp', it will chose the appropriate `limits`
          and `user_api` parameters for the specific use case of sequential BLAS
          calls within an OpenMP parallel region. The `user_api` parameter is
          ignored.

        - If None, this function does not do anything.

    user_api : {USER_APIS} or None (default=None)
        APIs of libraries to limit. Used only if `limits` is an int.

        - If "blas", it will only limit BLAS supported libraries ({BLAS_LIBS}).

        - If "openmp", it will only limit OpenMP supported libraries
          ({OPENMP_LIBS}). Note that it can affect the number of threads used
          by the BLAS libraries if they rely on OpenMP.

        - If None, this function will apply to all supported libraries.
    """

    def __init__(self, limits=None, user_api=None):
        super().__init__(ThreadpoolController(), limits=limits, user_api=user_api)

    @classmethod
    def wrap(cls, limits=None, user_api=None):
        return super().wrap(ThreadpoolController(), limits=limits, user_api=user_api)


class ThreadpoolController:
    """Collection of LibController objects for all loaded supported libraries

    Attributes
    ----------
    lib_controllers : list of `LibController` objects
        The list of library controllers of all loaded supported libraries.
    """

    # Cache for libc under POSIX and a few system libraries under Windows.
    # We use a class level cache instead of an instance level cache because
    # it's very unlikely that a shared library will be unloaded and reloaded
    # during the lifetime of a program.
    _system_libraries = dict()

    def __init__(self):
        self.lib_controllers = []
        self._load_libraries()
        self._warn_if_incompatible_openmp()

    @classmethod
    def _from_controllers(cls, lib_controllers):
        new_controller = cls.__new__(cls)
        new_controller.lib_controllers = lib_controllers
        return new_controller

    def info(self):
        """Return lib_controllers info as a list of dicts"""
        return [lib_controller.info() for lib_controller in self.lib_controllers]

    def select(self, **kwargs):
        """Return a ThreadpoolController containing a subset of its current
        library controllers

        It will select all libraries matching at least one pair (key, value) from kwargs
        where key is an entry of the library info dict (like "user_api", "internal_api",
        "prefix", ...) and value is the value or a list of acceptable values for that
        entry.

        For instance, `ThreadpoolController().select(internal_api=["blis", "openblas"])`
        will select all library controllers whose internal_api is either "blis" or
        "openblas".
        """
        for key, vals in kwargs.items():
            kwargs[key] = [vals] if not isinstance(vals, list) else vals

        lib_controllers = [
            lib_controller
            for lib_controller in self.lib_controllers
            if any(
                getattr(lib_controller, key, None) in vals
                for key, vals in kwargs.items()
            )
        ]

        return ThreadpoolController._from_controllers(lib_controllers)

    def _get_params_for_sequential_blas_under_openmp(self):
        """Return appropriate params to use for a sequential BLAS call in an OpenMP loop

        This function takes into account the unexpected behavior of OpenBLAS with the
        OpenMP threading layer.
        """
        if self.select(
            internal_api="openblas", threading_layer="openmp"
        ).lib_controllers:
            return {"limits": None, "user_api": None}
        return {"limits": 1, "user_api": "blas"}

    @_format_docstring(
        USER_APIS=", ".join('"{}"'.format(api) for api in _ALL_USER_APIS),
        BLAS_LIBS=", ".join(_ALL_BLAS_LIBRARIES),
        OPENMP_LIBS=", ".join(_ALL_OPENMP_LIBRARIES),
    )
    def limit(self, *, limits=None, user_api=None):
        """Change the maximal number of threads that can be used in thread pools.

        This function returns an object that can be used either as a callable (the
        construction of this object limits the number of threads) or as a context
        manager, in a `with` block to automatically restore the original state of the
        controlled libraries when exiting the block.

        Set the maximal number of threads that can be used in thread pools used in
        the supported libraries to `limits`. This function works for libraries that
        are already loaded in the interpreter and can be changed dynamically.

        This effect is global and impacts the whole Python process. There is no thread
        level isolation as these libraries do not offer thread-local APIs to configure
        the number of threads to use in nested parallel calls.

        Parameters
        ----------
        limits : int, dict, 'sequential_blas_under_openmp' or None (default=None)
            The maximal number of threads that can be used in thread pools

            - If int, sets the maximum number of threads to `limits` for each
              library selected by `user_api`.

            - If it is a dictionary `{{key: max_threads}}`, this function sets a
              custom maximum number of threads for each `key` which can be either a
              `user_api` or a `prefix` for a specific library.

            - If 'sequential_blas_under_openmp', it will chose the appropriate `limits`
              and `user_api` parameters for the specific use case of sequential BLAS
              calls within an OpenMP parallel region. The `user_api` parameter is
              ignored.

            - If None, this function does not do anything.

        user_api : {USER_APIS} or None (default=None)
            APIs of libraries to limit. Used only if `limits` is an int.

            - If "blas", it will only limit BLAS supported libraries ({BLAS_LIBS}).

            - If "openmp", it will only limit OpenMP supported libraries
              ({OPENMP_LIBS}). Note that it can affect the number of threads used
              by the BLAS libraries if they rely on OpenMP.

            - If None, this function will apply to all supported libraries.
        """
        return _ThreadpoolLimiter(self, limits=limits, user_api=user_api)

    @_format_docstring(
        USER_APIS=", ".join('"{}"'.format(api) for api in _ALL_USER_APIS),
        BLAS_LIBS=", ".join(_ALL_BLAS_LIBRARIES),
        OPENMP_LIBS=", ".join(_ALL_OPENMP_LIBRARIES),
    )
    def wrap(self, *, limits=None, user_api=None):
        """Change the maximal number of threads that can be used in thread pools.

        This function returns an object that can be used as a decorator.

        Set the maximal number of threads that can be used in thread pools used in
        the supported libraries to `limits`. This function works for libraries that
        are already loaded in the interpreter and can be changed dynamically.

        Parameters
        ----------
        limits : int, dict or None (default=None)
            The maximal number of threads that can be used in thread pools

            - If int, sets the maximum number of threads to `limits` for each
              library selected by `user_api`.

            - If it is a dictionary `{{key: max_threads}}`, this function sets a
              custom maximum number of threads for each `key` which can be either a
              `user_api` or a `prefix` for a specific library.

            - If None, this function does not do anything.

        user_api : {USER_APIS} or None (default=None)
            APIs of libraries to limit. Used only if `limits` is an int.

            - If "blas", it will only limit BLAS supported libraries ({BLAS_LIBS}).

            - If "openmp", it will only limit OpenMP supported libraries
              ({OPENMP_LIBS}). Note that it can affect the number of threads used
              by the BLAS libraries if they rely on OpenMP.

            - If None, this function will apply to all supported libraries.
        """
        return _ThreadpoolLimiter.wrap(self, limits=limits, user_api=user_api)

    def __len__(self):
        return len(self.lib_controllers)

    def _load_libraries(self):
        """Loop through loaded shared libraries and store the supported ones"""
        if sys.platform == "darwin":
            self._find_libraries_with_dyld()
        elif sys.platform == "win32":
            self._find_libraries_with_enum_process_module_ex()
        elif "pyodide" in sys.modules:
            self._find_libraries_pyodide()
        else:
            self._find_libraries_with_dl_iterate_phdr()

    def _find_libraries_with_dl_iterate_phdr(self):
        """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on POSIX system only.
        This code is adapted from code by Intel developer @anton-malakhov
        available at https://github.com/IntelPython/smp

        Copyright (c) 2017, Intel Corporation published under the BSD 3-Clause
        license
        """
        libc = self._get_libc()
        if not hasattr(libc, "dl_iterate_phdr"):  # pragma: no cover
            warnings.warn(
                "Could not find dl_iterate_phdr in the C standard library.",
                RuntimeWarning,
            )
            return []

        # Callback function for `dl_iterate_phdr` which is called for every
        # library loaded in the current process until it returns 1.
        def match_library_callback(info, size, data):
            # Get the path of the current library
            filepath = info.contents.dlpi_name
            if filepath:
                filepath = filepath.decode("utf-8")

                # Store the library controller if it is supported and selected
                self._make_controller_from_path(filepath)
