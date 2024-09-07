__all__ = ["OpenCV", "Raw", "Terminal"]

import importlib

class LazyLoader:
    def __init__(self, module_name, class_name):
        self._module_name = module_name
        self._class_name = class_name
        self._class_obj = None

    def _load(self):
        module = importlib.import_module(f'.{self._module_name}', package='spacecubes.io_devices')
        self._class_obj = getattr(module, self._class_name)

    def __getattr__(self, item):
        if self._class_obj is None:
            self._load()
        return getattr(self._class_obj, item)

    def __call__(self, *args, **kwargs):
        if self._class_obj is None:
            self._load()
        return self._class_obj(*args, **kwargs)

OpenCV = LazyLoader('opencv', 'OpenCV')
Raw = LazyLoader('raw', 'Raw')
Terminal = LazyLoader('terminal', 'Terminal')
