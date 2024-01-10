"""
@User:  
        将@Module{meters.py}中定义的@Class{MeterDict}再次聚合, 支持对特定场景进行多meters的记录, 如train, valid
        
        A standalone module for aggregating metrics.

        Metrics can be logged from anywhere using the @Funcs{log_*} defined in this module. 
        The logged values will be aggregated dynamically based on the aggregation context in which the logging occurs.
        See the @Func{aggregate} context manager for more details.
        @Question: 这个函数其实没看懂
"""



"""
@Question:  What's @Pkg{comtextlib}?
@Answer: 详见: https://docs.python.org/3/library/contextlib.html
"""
import contextlib
import uuid
from collections import defaultdict
from typing import Callable, List, Optional

from .meters import *



# Aggregation contexts are considered "active" when inside the scope
# created by the @Func{aggregate} context manager.
_aggregators = OrderedDict()
_active_aggregators = OrderedDict()
#@Callback: defaultdict, 第一个参数为default_factory属性提供初始值, 默认为None，可用于Value的初始化
_active_aggregators_cnt = defaultdict(lambda: 0)



"""@Desc: Reset all metrics aggregators."""
def reset() -> None:
    _aggregators.clear()
    _active_aggregators.clear()
    _active_aggregators_cnt.clear()

    # The "default" aggregator observes all logged values.
    _aggregators["default"] = MetersDict()
    _active_aggregators["default"] = _aggregators["default"]
    _active_aggregators_cnt["default"] = 1


reset()

"""
@Desc: Context manager to aggregate metrics under a given name.
@param{name}:   @Type{str}, name of the aggregation. Defaults to a random/temporary name if not given explicitly.
    aggregation contexts are uniquely identified by their @Param{name}(e.g., train, valid)
    Creating a context with an existing name will reuse the corresponding @Class{MeterDict} instance.
    If no name is given, then a temporary aggregator will be created.
@Param{new_root}:   @Type{bool}, make this aggregation the root of a new aggregation stack.
    Aggregations can be nested. 
    If @Param{new_root} is @Cont{False}, then logged metrics will be recorded along the entire stack of nested aggregators, including a global "default" aggregator. 
    If @Param{new_root} is @Cont{True}, then this aggregator will be the root of a new aggregation stack, thus bypassing any parent aggregators.
@Usage: @Code{
    with metrics.aggregate("train"):    #@Param{name} is "train"
        for step, batch in enumerate(epoch):
            with metrics.aggregate("train_inner") as agg:
                metrics.log_scalar("loss", get_loss(batch))
                if step % log_interval == 0:
                    print(agg.get_smoothed_value("loss"))
                    agg.reset()
    print(metrics.get_smoothed_values("train")["loss"])
} 
"""
@contextlib.contextmanager
def aggregate(name: Optional[str] = None, new_root: bool = False):
    if name is None:
        #@Explain: generate a temporary name
        name = str(uuid.uuid4())
        assert name not in _aggregators
        agg = MetersDict()
    else:
        assert name != "default"
        #@Callback: @Func{setdefault} is similar to @Func{get}, 如果dict中没有key，将会添加key，并将值设置为default value, Return: key相对应的value
        agg = _aggregators.setdefault(name, MetersDict())
        #@Now: Creating a context with an existing name will reuse the corresponding @Class{MeterDict} instance.

    if new_root:
        backup_aggregators = _active_aggregators.copy()
        _active_aggregators.clear()
        backup_aggregators_cnt = _active_aggregators_cnt.copy()
        _active_aggregators_cnt.clear()

    _active_aggregators[name] = agg
    _active_aggregators_cnt[name] += 1

    yield agg
    
    #@Now: @Statement{with} ends
    _active_aggregators_cnt[name] -= 1
    if _active_aggregators_cnt[name] == 0 and name in _active_aggregators:
        del _active_aggregators[name]

    if new_root:
        _active_aggregators.clear()
        _active_aggregators.update(backup_aggregators)
        _active_aggregators_cnt.clear()
        _active_aggregators_cnt.update(backup_aggregators_cnt)



def get_active_aggregators() -> List[MetersDict]:
    return list(_active_aggregators.values())



"""
@Desc: Log a scalar value.
@Param{key}:    @Type{str}, name of the field to log.
@Param{value}:  @Type{float}, value to log.
@Param{weight}: @Type{float}, weight that this value contributes to the average. A weight of 0 will always log the latest value.
@Param{priority}:@Type{int}, smaller values are logged earlier in the output.
@Param{round}:  @Type{Optional[int]}, number of digits to round to when displaying.
"""
def log_scalar(
    key: str,
    value: float,
    weight: float = 1,
    priority: int = 10,
    round: Optional[int] = None,
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, AverageMeter(round=round), priority)
        agg[key].update(value, weight)



"""
@Desc: Log a scalar value that is summed for reporting.
@Param{key}:    @Type{str}, name of the field to log.
@Param{value}:  @Type{float}, value to log.
@Param{priority}:@Type{int}, smaller values are logged earlier in the output.
@Param{round}:  @Type{Optional[int]}, number of digits to round to when displaying.
"""
def log_scalar_sum(
    key: str,
    value: float,
    priority: int = 10,
    round: Optional[int] = None,
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, SumMeter(round=round), priority)
        agg[key].update(value)



"""
@Desc: Log scalar values that is concated for reporting.
@Param{key}:    @Type{str}, name of the field to log.
@Param{value}:  @Type{float}, value to log.
@Param{priority}:@Type{int}, smaller values are logged earlier in the output.
@Param{round}:  @Type{Optional[int]}, number of digits to round to when displaying.
"""
def log_concat_tensor(
    key: str,
    value: torch.Tensor,
    priority: int = 10,
    dim: int = 0,
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, ConcatTensorMeter(dim=dim), priority)
        agg[key].update(value)



"""
@Desc: Log a scalar value derived from other meters.
@Param{key}:    @Type{str}, name of the field to log.
@Param{fn}:     @Type{Callable[[MetersDict], float]}, function that takes a single argument {meters} and returns the derived value.
@Param{priority}:@Type{int}, smaller values are logged earlier in the output.
"""
def log_derived(key: str, fn: Callable[[MetersDict], float], priority: int = 20):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, MetersDict._DerivedMeter(fn), priority)



"""
@Desc: Log the rate of some quantity per second.
@Param{key}:    @Type{str}, name of the field to log.
@Param{value}:  @Type{float}, value to log.
@Param{priority}:@Type{int}, smaller values are logged earlier in the output.
@Param{round}:  @Type{Optional[int]}, number of digits to round to when displaying.
"""
def log_speed(
    key: str,
    value: float,
    priority: int = 30,
    round: Optional[int] = None,
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, TimeMeter(round=round), priority)
            agg[key].reset()
            #@Now: reset meter on the first call, It has started!
        else:
            agg[key].update(value)



"""
@Desc: Log the duration of some event in seconds. The duration will be computed once @Func{log_stop_time} is called.
@Param{key}:    @Type{str}, name of the field to log.
@Param{priority}:@Type{int}, smaller values are logged earlier in the output.
@Param{round}:  @Type{Optional[int]}, number of digits to round to when displaying.
"""
def log_start_time(key: str, priority: int = 40, round: Optional[int] = None):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, StopwatchMeter(round=round), priority)
        agg[key].start()
        #@Now: The @Class{StopwatchMeter} instace has started!



"""
@Desc: Log the duration of some event in seconds. The duration will be computed since @Func{log_start_time} was called.
@Note: Set weight > 0 to report the average time instead of the sum.
@Param{key}:    @Type{str}, name of the field to log.
@Param{weight}: @Type{float}, weight that this time contributes to the average.
@Param{prehook}:@Type{function, no arguments}, will be called before the timer is stopped.
                @Example, use @Code{prehook=torch.cuda.synchronize} to make sure all gpu operations are done before timer is stopped.
"""
def log_stop_time(key: str, weight: float = 0.0, prehook=None):
    for agg in get_active_aggregators():
        if key in agg:
            agg[key].stop(weight, prehook)



"""
@Desc: Log using a custom Meter. Any extra @Param{*args} or @Param{**kwargs} will be passed through to the Meter's @Func{update} method.
可以通过继承@Class{BaseMeter}定义新的测度，并进行记录
@Param{new_meter_fn}:@Type{Callable[[], BaseMeter]}, function that returns a new Meter instance.
@Param{key}:    @Type{str}, name of the field to log
@Param{priority}:    @Type{int}, smaller values are logged earlier in the output.
"""
def log_custom(
    new_meter_fn: Callable[[], BaseMeter],
    key: str,
    *args,
    priority: int = 50,
    **kwargs,
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, new_meter_fn(), priority)
        agg[key].update(*args, **kwargs)


"""
@Desc: Reset Meter instance aggregated under a given @Param{name} and @Param{key}.
@Param{name}: 指定具体的aggregator
@Param{key}: 指定具体的meter
"""
def reset_meter(name: str, key: str) -> None:
    meter = get_meter(name, key)
    if meter is not None:
        meter.reset()



"""
@Desc: Reset Meter instances aggregated under a given @Param{name}.
"""
def reset_meters(name: str) -> None:
    meters = get_meters(name)
    if meters is not None:
        meters.reset()



"""
@Desc: Get a single Meter instance aggregated under @Param{name} and @Param{key}.
@Return: Meter or None if no metrics have been logged under @Param{name} and @Param{key}.
"""
def get_meter(name: str, key: str) -> BaseMeter:
    if name not in _aggregators:
        return None
    return _aggregators[name].get(key, None)



"""
@Desc: Get Meter instances aggregated under a given @Param{name}.
@Return: MetersDict or None if no metrics have been logged under @Param{name}.
"""
def get_meters(name: str) -> MetersDict:
    return _aggregators.get(name, None)



"""
@Desc: Get a single smoothed value.
@Excection: @KeyError{if no metrics have been logged under @Param{name} and @Param{key}}.
"""
def get_smoothed_value(name: str, key: str) -> float:
    return _aggregators[name].get_smoothed_value(key)



"""
@Desc: Get smoothed values aggregated under a given @Param{name}.
@Exception: @KeyError{if no metrics have been logged under @Parm{name}.
"""
def get_smoothed_values(name: str) -> Dict[str, float]:
    return _aggregators[name].get_smoothed_values()



def state_dict():
    return OrderedDict([(name, agg.state_dict()) for name, agg in _aggregators.items()])



def load_state_dict(state_dict):
    for name, agg_state in state_dict.items():
        _aggregators[name] = MetersDict()
        _aggregators[name].load_state_dict(agg_state)


def xla_metrics_report():
    try:
        import torch_xla.debug.metrics as met

        print(met.metrics_report())
    except ImportError:
        return
