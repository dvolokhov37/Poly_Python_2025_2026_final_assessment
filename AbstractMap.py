from abc import ABC, abstractmethod
import os

from SimpleMap import SimpleMap


class AbstractMap(ABC):

    def __init__(self, path: str | list[str]):
        self.path = os.path.abspath(path) if isinstance(path, str) else [os.path.abspath(p) for p in path]
        self.map: SimpleMap | list[SimpleMap] | None = None

    @abstractmethod
    def load(self) -> SimpleMap | list[SimpleMap]:
        pass

    @abstractmethod
    def read_file(self):
        pass

    @abstractmethod
    def preprocessing(self, *args, **kwargs):
        pass
