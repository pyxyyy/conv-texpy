"""
Utilities
"""
from collections import defaultdict

import heapq
import json
import logging
from typing import Dict, TypeVar, Tuple, NamedTuple, List, Union, Any, cast
from typing import Sequence, Callable
from .quality_control import QualityControlDecision


T = TypeVar("T")

# Ideally, we would use # the "SimpleObject" instead of "Any" in the
# type definition below, but sadly, MyPy does not yet support recursive
# types.
SimpleObject = Union[bool, int, str, List[Any], Dict[str, Any]]


logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o: object) -> Dict:
        if isinstance(o, QualityControlDecision):
            return {
                    "_type": "QualityControlDecision",
                    "_fields": o.asdict(),
            }
        else:
            return super().default(o)

def custom_decoder_object_hook(o: object):
    if isinstance(o, dict) and "_type" in o:
        if o["_type"] == "QualityControlDecision" and "_fields" in o:
            return QualityControlDecision(**o["_fields"])
    return o

def JsonFile(*args, **kwargs):
    def _ret(fname):
        return json.load(open(fname, *args, **kwargs), object_hook=custom_decoder_object_hook)
    return _ret

def load_jsonl(fstream) -> List[SimpleObject]:
    if isinstance(fstream, str):
        with open(fstream) as fstream_:
            return load_jsonl(fstream_)

    return [json.loads(line, object_hook=custom_decoder_object_hook) for line in fstream]

def save_jsonl(fstream, objs: List[SimpleObject]):
    if isinstance(fstream, str):
        with open(fstream, "w") as fstream_:
            save_jsonl(fstream_, objs)
        return

    for obj in objs:
        fstream.write(json.dumps(obj, sort_keys=True, cls=CustomJSONEncoder))
        fstream.write("\n")

def first(itable):
    try:
        return next(iter(itable))
    except StopIteration:
        return None

def force_user_input(prompt, options):
    ret = None
    while ret is None:
        ret = input(prompt + "|".join(options) + ": ").strip()
        if ret.lower() not in options:
            ret = None
    return ret

def obj_diff(obj, obj_):
    ret = True
    for k in obj:
        if k not in obj_:
            print("Key {} missing in arg2".format(k))
            ret = False
        if obj[k] != obj_[k]:
            if isinstance(obj[k], dict) and isinstance(obj_[k], dict):
                obj_diff(obj[k], obj_[k])
                ret = False
            else:
                print("{}: args1 has {}, args2 has {}".format(k, obj[k], obj_[k]))
    for k in obj_:
        if k not in obj:
            print("Key {} missing in arg1".format(k))
            ret = False
    return ret


def sanitize(obj: T) -> T:
    """
    Sanitize an object containing dictionaries by removing any entries
    with a key that starts with '_'.
    """
    if isinstance(obj, list):
        return cast(T, [sanitize(obj_) for obj_ in obj])
    elif isinstance(obj, dict):
        return cast(T, {key: sanitize(value) for key, value in obj.items() if not key.startswith("_")})
    else:
        return obj


def mark_for_sanitization(obj: T, field_names: List[str]) -> T:
    """
    Updates fields in `obj` by prepending a '_' for all fields in `field_names`.
    These fields are subsequently removed by `sanitize` when sending data to
    MTurk.
    """
    if isinstance(obj, list):
        return cast(T, [mark_for_sanitization(obj_) for obj_ in obj])
    elif isinstance(obj, dict):
        return cast(T, {
            f"_{key}" if key in field_names else key: sanitize(value)
            for key, value in obj.items()})
    else:
        return obj


def unmark_for_sanitization(obj: T) -> T:
    """
    Updates fields in `obj` by removing any '_' prefixes.
    This undoes the transformation in `mark_for_sanitization`.
    """
    if isinstance(obj, list):
        return cast(T, [unmark_for_sanitization(obj_) for obj_ in obj])
    elif isinstance(obj, dict):
        return cast(T, {
            (key[1:] if key.startswith("_") else key): unmark_for_sanitization(value)
            for key, value in obj.items()})
    else:
        return obj


# region: Span utilities
Span = Tuple[int, int]


class WeightedSpan(NamedTuple):
    """
    A span tuple with a weight / count field
    """
    begin: int
    end: int
    weight: int = 1

    def __contains__(self, other):
        return other.begin >= self.begin and other.end <= self.end

    @classmethod
    def collapse_spans(cls, lst: List['WeightedSpan']) -> List['WeightedSpan']:
        """
        Convert a list of spans into non-overlapping versions with weights
        for each overlapping section.
        """
        if not lst: return []

        all_spans = list(lst)
        heapq.heapify(all_spans)

        # 1. Figure out what the interval spans that we'll count over are
        #    We do this by setting up split points
        canonical_spans = [heapq.heappop(all_spans)]
        while all_spans:
            span = heapq.heappop(all_spans)
            last_span = canonical_spans[-1]
            assert last_span.begin <= span.begin

            # If the spans don't even overlap, we can safely add this to
            # the canonical list.
            if not(last_span.begin < span.end and span.begin < last_span.end):
                canonical_spans.append(span)
            # We now handle the different overlapping cases.
            elif last_span.begin < span.begin:
                # We are going to split last_span and span into two segments
                # each (with one overlapping span) pivoted at span.begin
                # First, we'll update last_span to its new boundary.
                canonical_spans[-1] = WeightedSpan(last_span.begin, span.begin, last_span.weight)
                # Then, we'll break last_span at span.begin
                heapq.heappush(all_spans, 
                        WeightedSpan(span.begin, last_span.end, last_span.weight))
                # And push `span` back into the queue.
                heapq.heappush(all_spans, 
                        WeightedSpan(span.begin, span.end, span.weight))
            elif last_span.end < span.end:
                # We are going to split span into two segments pivoted
                # around last_span.end, and increment counts appropriately
                canonical_spans[-1] = WeightedSpan(last_span.begin, last_span.end,
                        last_span.weight + span.weight)
                # Create a new segment from [last_span.end, span.end)
                heapq.heappush(all_spans,
                        WeightedSpan(last_span.end, span.end, span.weight))
            else:
                # We have a complete overlap and are just going to increment
                # counts
                canonical_spans[-1] = WeightedSpan(last_span.begin, last_span.end,
                        last_span.weight + span.weight)

        return canonical_spans


def collapse_spans(lst: List[Span]) -> List[Span]:
    """
    Convert a list of spans into non-overlapping versions with weights
    for each overlapping section.
    """
    return [(wspan.begin, wspan.end)
            for wspan in WeightedSpan.collapse_spans([WeightedSpan(*span) for span in lst])]


def test_collapse_spans():
    # Disjoint spans
    assert collapse_spans([(10, 20), (30, 40)]) == [
            WeightedSpan(10, 20, 1), WeightedSpan(30, 40, 1)] 
    # Overlapping spans
    assert collapse_spans([(10, 30), (20, 40)]) == [
            WeightedSpan(10, 20, 1), WeightedSpan(20, 30, 2), WeightedSpan(30, 40, 1)] 

    # More complex overlap
    assert collapse_spans([(33, 56), (38, 41), (45, 48), (52, 56)]) == [
            WeightedSpan(33, 38, 1),
            WeightedSpan(38, 41, 2),
            WeightedSpan(41, 45, 1),
            WeightedSpan(45, 48, 2),
            WeightedSpan(48, 52, 1),
            WeightedSpan(52, 56, 2),
            ]


def merge_adjacent_spans(spans: List[Span]) -> List[Span]:
    """
    Merges adjacent spans in a list of spans.
    
    Usage:
    >>> ret = merge_adjacent_spans([(0, 10), (10, 20), (30, 40)])
    >>> assert ret == [(0, 20), (30, 40)]

    Args:
        spans: a list of non-overlapping spans, some of which may be adjacent

    Returns:
        a list of non-overlapping, non-adjacent spans equivalent to `spans`.
    """
    # 4. Collapse adjacent intervals in reverse
    for i in range(len(spans) - 1, 0, -1):
        span, prev_span = spans[i], spans[i - 1]
        if prev_span[1] == span[0]:
            spans.pop(i)
            spans[i - 1] = (prev_span[0], span[1])
    return spans


def test_merge_adjacent_spans():
    assert [(0, 4), (8, 18)] == merge_adjacent_spans([
        (0, 4), (8, 18)
    ])

    assert [(0, 6), (8, 18)] == merge_adjacent_spans([
        (0, 4), (4, 6), (8, 18)
    ])

    assert [(0, 18)] == merge_adjacent_spans([
        (0, 4), (4, 8), (8, 18)
    ])
# endregion


# region: dictionary manipulation
W = TypeVar('W')


def invert_dict(data: Dict[T, Dict[W, Any]]) -> Dict[W, Dict[T, Any]]:
    ret = defaultdict(dict)
    for m, dct in data.items():
        for n, v in dct.items():
            ret[n][m] = v
    return ret


def flatten_dict(data: Dict[T, Dict[W, Any]]) -> Dict[Tuple[T, W], Any]:
    ret = {}
    for k, vs in data.items():
        for k_, v in vs.items():
            ret[k, k_] = v
    return ret


def group_by(data: Sequence[T], key_fn: Callable[[T], W]) -> Dict[W, List[T]]:
    """
    Groups a sequence of elements by some key

    Usage:
        >>> groups = group_by([("a", 1), ("b", 2), ("a", 3), ("b", 4)], lambda elem: elem[0])
        >>> assert groups == {"a": [("a", 1), ("a", 3)], "b": [("b", 2), , ("b", 4)]}

    Args:
        data: A sequence of objects
        key_fn: A function that produces a key given an object from `data`

    Returns:
        A dictionary with keys returned by `key_fn`; each value is a list of objects from `data`
        that have the same key.
    """
    ret: Dict[W, List[T]] = defaultdict(list)
    for elem in data:
        ret[key_fn(elem)].append(elem)
    return ret

# endregion
