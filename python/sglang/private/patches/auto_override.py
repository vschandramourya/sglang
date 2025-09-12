import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import threading
from contextlib import contextmanager
from types import ModuleType

PRIVATE_PREFIX: str = "sglang.private"
TARGET_PREFIX: str = "sglang.srt"


def safe_find_spec(fullname: str):
    try:
        return importlib.util.find_spec(fullname)
    except Exception:
        return None


def apply_patch(public_mod: ModuleType, private_mod: ModuleType):
    """
    Default patching semantics:
    - If private_mod defines __apply_patch__(public_mod), call it and return.
    - Else copy all public attributes (not starting with '_') from private_mod
      onto public_mod, except those listed in __patch_exclude__; or only those
      listed in __patch_include__ if provided.
    - Merge __all__ if present.
    - Idempotent via a guard flag on public_mod.
    """
    if getattr(public_mod, "__private_patched__", False):
        return

    # 1) allow custom patch function
    fn = getattr(private_mod, "__apply_patch__", None)
    if callable(fn):
        fn(public_mod)
        setattr(public_mod, "__private_patched__", True)
        return

    # 2) default: name-based copy
    include = set(getattr(private_mod, "__patch_include__", []) or [])
    exclude = set(getattr(private_mod, "__patch_exclude__", []) or [])

    if include:
        names = include
    else:
        names = {n for n in dir(private_mod) if not n.startswith("_")}

    names -= exclude

    for name in names:
        setattr(public_mod, name, getattr(private_mod, name))

    # Merge __all__
    if hasattr(public_mod, "__all__") or hasattr(private_mod, "__all__"):
        pub_all = set(getattr(public_mod, "__all__", []) or [])
        prv_all = set(getattr(private_mod, "__all__", []) or [])
        # if using default mode, also expose copied names
        pub_all |= prv_all | set(names)
        public_mod.__all__ = sorted(pub_all)

    setattr(public_mod, "__private_patched__", True)


@contextmanager
def bypass():
    finder = PATCHING_FINDER
    finder.tls.bypass = getattr(finder.tls, "bypass", 0) + 1
    try:
        yield
    finally:
        finder.tls.bypass -= 1


class PatchingFinder(importlib.abc.MetaPathFinder):
    """
    Post-load patcher:
      - If a private twin exists for 'sglang.srt.*', wrap the *upstream* loader.
      - Load upstream normally, then import the private twin and monkey patch
        the public module object.
      - Never replaces packages; it only patches module objects after load.
    """

    def __init__(self):
        self.tls = threading.local()
        self.tls.bypass = 0  # hard bypass

    def find_spec(self, fullname, path=None, target=None):

        # Only handle sglang.srt and its submodules
        if fullname != TARGET_PREFIX and not fullname.startswith(TARGET_PREFIX + "."):
            return None

        if getattr(self.tls, "bypass", 0):
            return None

        private_name = fullname.replace(TARGET_PREFIX, PRIVATE_PREFIX, 1)

        # Do we have a private twin? (module or package)
        with bypass():
            private_spec = safe_find_spec(private_name)

        if private_spec is None:
            return None  # nothing to patch → let default importers handle it

        # Find the real upstream spec for the public name
        with bypass():
            upstream_spec = safe_find_spec(fullname)

        if upstream_spec is None or upstream_spec.loader is None:
            return None  # shouldn't happen; fallback to default behavior

        # Wrap the upstream loader
        upstream_loader = upstream_spec.loader
        private_name_str = private_name  # close over value

        class WrapperLoader(importlib.abc.Loader):
            def create_module(self, spec):
                # Delegate creation to upstream loader if it implements it
                if hasattr(upstream_loader, "create_module"):
                    return upstream_loader.create_module(upstream_spec)
                return None  # default creation

            def exec_module(self, module: ModuleType):
                # 1) Execute upstream (module is already registered in sys.modules)
                upstream_loader.exec_module(module)
                # 2) Import private twin and patch the *public* module object
                with bypass():
                    prv = importlib.import_module(private_name_str)
                apply_patch(module, prv)

        # Return the upstream spec but with our wrapper loader
        upstream_spec.loader = WrapperLoader()
        return upstream_spec


PATCHING_FINDER = PatchingFinder()


def enable_private_overrides():
    """
    Install the patching finder at the front of sys.meta_path (idempotent).
    Call this before any imports of 'sglang.srt.*'.
    """
    if not any(isinstance(f, PatchingFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, PATCHING_FINDER)


enable_private_overrides()
