from typing import Iterable, TypeVar, Union, Any
import io

X = TypeVar('X')

def tqdm(
        iterable: Iterable[X]=None,
        desc: str=None,
        total: int=None,
        leave: bool=True,
        file: Union[io.TextIOWrapper, io.StringIO] = None,
        ncols: int = None,
        mininterval: float =0.1,
        maxinterval: float =10.0,
        miniters: int = None,
        ascii: bool=None,
        disable: bool=False,
        unit: str='it',
        unit_scale: Union[bool, int, float]=False,
        dynamic_ncols: bool=False,
        smoothing: float=0.3,
        bar_format: str=None,
        initial: int=0,
        position: int=None,
        postfix: Any=None,
        unit_divisor: float=1000) -> Iterable[X]:
    ...

def trange(
        count: int,
        desc: str=None,
        total: int=None,
        leave: bool=True,
        file: Union[io.TextIOWrapper, io.StringIO] = None,
        ncols: int = None,
        mininterval: float =0.1,
        maxinterval: float =10.0,
        miniters: int = None,
        ascii: bool=None,
        disable: bool=False,
        unit: str='it',
        unit_scale: Union[bool, int, float]=False,
        dynamic_ncols: bool=False,
        smoothing: float=0.3,
        bar_format: str=None,
        initial: int=0,
        position: int=None,
        postfix: Any=None,
        unit_divisor: float=1000) -> Iterable[int]:
    ...
