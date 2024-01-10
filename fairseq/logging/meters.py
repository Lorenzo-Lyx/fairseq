"""
@User: This file contains numerous meters, including meters for time statistics. 
       The base class for these meters is BaseMeter. 
       Additionally, an ordered meters dictionary has been implemented, which is not intended for reassignment, 
       and its efficiency might be relatively low.
"""



"""
@Question: What is @Pkg{bisect}?
@Answer:   这是Python3的二分查找库;
           @Func{insort}插入函数
           @Func{bisect}(@Param{ls}(从小到大), @Param(x), hi, key) -> @Param{x}: 返回大于x的第一个索引
           @Func{bisect_right}：与@Func{bisect}相同
           @Func{bisect_left}: 返回大于等于x的第一个索引
"""
import bisect
import time
from collections import OrderedDict
from typing import Dict, Optional



try:
    import torch

    def type_as(a, b):
        if torch.is_tensor(a) and torch.is_tensor(b):
            return a.to(b)
        else:
            return a

except ImportError:
    torch = None

    def type_as(a, b):
        return a



try:
    import numpy as np
except ImportError:
    np = None



"""@Desc: Base class for Meters."""
class BaseMeter(object):
    def __init__(self):
        #@Callback: @Keyword{pass}是一个空语句，保持程序结构的完整性，不做任何事情，仅仅占位
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def reset(self):
        raise NotImplementedError

    #@Desc: Smoothed value used for logging.
    #@Callback{property}: 创建只读属性，将方法转换为相同名称的只读属性
    #                     这也是设置私有属性的方法，通过_隐藏属性名，然后使用property修饰的方法创建只读属性
    #                     用户在使用的时候无法随意修改
    @property
    def smoothed_value(self) -> float:
        raise NotImplementedError



#@Desc: round the @Param{number} in @Param{ndigits} safely.
#@Question: Why safe? 
def safe_round(number, ndigits):
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
        return safe_round(number.item(), ndigits)
    else:
        return number


"""@Desc: Computes and stores the average and current value"""
class AverageMeter(BaseMeter):
    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.reset()

    def reset(self):
        #@Explain: The value most recent update.
        self.val = None
        #@Explain: The sum from all updates
        self.sum = 0
        #@Explain: total n from all updates
        self.count = 0

    #@Desc: update @Param{n} @Param{val}s
    #@Param{n} may be @Type{float}.
    def update(self, val, n=1):
        if val is not None:
            self.val = val
            if n > 0:
                self.sum = type_as(self.sum, val) + (val * n)
                self.count = type_as(self.count, n) + n

    def state_dict(self):
        return {
            "val": self.val,
            "sum": self.sum,
            "count": self.count,
            "round": self.round,
        }

    def load_state_dict(self, state_dict):
        self.val = state_dict["val"]
        self.sum = state_dict["sum"]
        self.count = state_dict["count"]
        self.round = state_dict.get("round", None)

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else self.val

    #@Desc: 与@Func{avg}相比，增加了@Func{safe_round}操作
    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val



"""@Desc: Computes and stores the sum"""
class SumMeter(BaseMeter):
    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.reset()

    def reset(self):
        self.sum = 0  # sum from all updates

    def update(self, val):
        if val is not None:
            self.sum = type_as(self.sum, val) + val

    def state_dict(self):
        return {
            "sum": self.sum,
            "round": self.round,
        }

    def load_state_dict(self, state_dict):
        self.sum = state_dict["sum"]
        self.round = state_dict.get("round", None)

    @property
    def smoothed_value(self) -> float:
        val = self.sum
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val



"""@Desc: Concatenates tensors"""
class ConcatTensorMeter(BaseMeter):
    def __init__(self, dim=0):
        super().__init__()
        self.reset()
        self.dim = dim

    def reset(self):
        self.tensor = None

    def update(self, val):
        if self.tensor is None:
            self.tensor = val
        else:
            self.tensor = torch.cat([self.tensor, val], dim=self.dim)

    def state_dict(self):
        return {
            "tensor": self.tensor,
        }

    def load_state_dict(self, state_dict):
        self.tensor = state_dict["tensor"]

    #@Return: a dummy value
    @property
    def smoothed_value(self) -> float:
        return []



"""@Desc: Computes the average occurrence of some event per second"""
class TimeMeter(BaseMeter):
    def __init__(
        self,
        init: int = 0,
        n: int = 0,
        round: Optional[int] = None,
    ):
        self.round = round
        self.reset(init, n)

    def reset(self, init=0, n=0):
        #@Explain: @Var{self.init} is the initial time.
        self.init = init
        self.start = time.perf_counter()
        #@Explain: @Var{self.n} is the number of occureence of some event.
        self.n = n
        #@Explain: @Var{self.i} is the number of updation.
        self.i = 0

    def update(self, val=1):
        self.n = type_as(self.n, val) + val
        self.i += 1

    def state_dict(self):
        return {
            "init": self.elapsed_time,
            "n": self.n,
            "round": self.round,
        }

    def load_state_dict(self, state_dict):
        if "start" in state_dict:
            # backwards compatibility for old state_dicts
            self.reset(init=state_dict["init"])
        else:
            self.reset(init=state_dict["init"], n=state_dict["n"])
            self.round = state_dict.get("round", None)

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.perf_counter() - self.start)

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val



"""@Desc: Computes the sum/avg duration of some event in seconds"""
class StopwatchMeter(BaseMeter):
    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.sum = 0
        self.n = 0
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    #@User: 可以不断调用这个函数，同时对多个event进行计时，最终可以计算他们的平均时间/总时间
    #@Question: What is @Param{prehook}?
    def stop(self, n=1, prehook=None):
        if self.start_time is not None:
            if prehook is not None:
                prehook()
            delta = time.perf_counter() - self.start_time
            self.sum = self.sum + delta
            self.n = type_as(self.n, n) + n

    def reset(self):
        #@Explain: @Var{self.sum} is the cumulative time during which stopwatch was active
        self.sum = 0
        #@Explain: @Var{self.n} is the total n across all start/stop
        self.n = 0
        self.start()

    def state_dict(self):
        return {
            "sum": self.sum,
            "n": self.n,
            "round": self.round,
        }

    def load_state_dict(self, state_dict):
        self.sum = state_dict["sum"]
        self.n = state_dict["n"]
        self.start_time = None
        self.round = state_dict.get("round", None)

    #@Return: self.n > 0 to report the average time instead of the sum. It @Depend{@Func{stop}}
    @property
    def avg(self):
        return self.sum / self.n if self.n > 0 else self.sum

    @property
    def elapsed_time(self):
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    @property
    def smoothed_value(self) -> float:
        #@Explain: 如果只开始，不停止，@Self.sum无法更新
        val = self.avg if self.sum > 0 else self.elapsed_time
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val



"""
@Desc:  A sorted dictionary of @Class{Meters}.
        Meters are sorted according to a priority that is given when the
        meter is first added to the dictionary.
"""
class MetersDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #@Explain: @Var{self.priorities} is like: [(priority, some integer, key)...], @Depend{@Func{__setitem__}}
        self.priorities = []

    #@Param{value}: @Tuple{(priority, meter instance)}
    def __setitem__(self, key, value):
        assert key not in self, "MetersDict doesn't support reassignment"
        priority, value = value
        #@Now: @Var{value} is an instance of some Meter.
        bisect.insort(self.priorities, (priority, len(self.priorities), key))
        super().__setitem__(key, value)
        #@Explain: reorder dict to match priorities
        for _, _, key in self.priorities:
            self.move_to_end(key)

    def add_meter(self, key, meter, priority):
        self.__setitem__(key, (priority, meter))

    #@Return: @List{[(priority, key, The class name of Meter, this meter's state_dict)...]}
    def state_dict(self):
        return [
            (pri, key, self[key].__class__.__name__, self[key].state_dict())
            for pri, _, key in self.priorities
            # can't serialize DerivedMeter instances
            if not isinstance(self[key], MetersDict._DerivedMeter)
        ]

    def load_state_dict(self, state_dict):
        self.clear()
        self.priorities.clear()
        #@Explain: 通过globals()获取确据定义域dict，查找到相应的类，构建一个实例 =>0 加载状态 => 重建dict
        for pri, key, meter_cls, meter_state in state_dict:
            meter = globals()[meter_cls]()
            meter.load_state_dict(meter_state)
            self.add_meter(key, meter, pri)

    #@Desc: Get a single smoothed value.
    def get_smoothed_value(self, key: str) -> float:
        meter = self[key]
        if isinstance(meter, MetersDict._DerivedMeter):
            return meter.fn(self)
        else:
            return meter.smoothed_value

    #@Desc: Get all smoothed values in orderedDict.
    def get_smoothed_values(self) -> Dict[str, float]:
        return OrderedDict(
            [
                (key, self.get_smoothed_value(key))
                for key in self.keys()
                if not key.startswith("_")
            ]
        )

    #@Desc: Reset Meter instances.
    def reset(self):
        for meter in self.values():
            if isinstance(meter, MetersDict._DerivedMeter):
                continue
            meter.reset()

    #@Desc: A Meter whose values are derived from other Meters.
    class _DerivedMeter(BaseMeter):
        def __init__(self, fn):
            self.fn = fn

        def reset(self):
            pass