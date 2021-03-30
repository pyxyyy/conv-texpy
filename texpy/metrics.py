from collections import defaultdict
from itertools import chain

import numpy as np
import scipy.stats as scstats
from typing import List, TypeVar, Dict, Any, Tuple, Optional, Iterable, Callable
from .aggregators import mean, std, median, percentile, median_absolute_deviation
from .util import Span, WeightedSpan, collapse_spans, invert_dict, flatten_dict

T = TypeVar('T')
W = TypeVar('W')


# region: dictionary manipulation
def as_task_worker_dict(values: Iterable[Tuple[T, W, Any]]) -> Dict[T, Dict[W, Any]]:
    ret = defaultdict(dict)
    for id_1, id_2, value in values:
        ret[id_1][id_2] = value
    return ret
# endregion


# distributional metrics
def distribution(prefix: str, values: List[float]) -> Dict[str, float]:
    """
    Returns some distributional metrics on @values. In particular we generate:
    {prefix}.mean - the mean of @values
    {prefix}.std  - the standard deviation of the @values
    {prefix}.p05  - the 5th percentile of @values
    {prefix}.p50  - the median of @values
    {prefix}.p95  - the 95th percentile of @values

    :param prefix: How to prefix the returned values.
    :param values: A list of values.
    :return: A dictionary with the above metrics.
    """
    return {
        f"{prefix}.mean": mean(values),
        f"{prefix}.std":  std(values),
        f"{prefix}.p05":  percentile(values, 5),
        f"{prefix}.p50":  median(values),
        f"{prefix}.p95":  percentile(values, 95),
    }


def modified_z_score(prefix: str, values: List[float], cutoff: float = 3.5) -> Dict[str, float]:
    """
    Returns modified z score parameters for @values. Useful to identify outliers.

    The z-score can be computed for a value 'y' using
        z = 0.6745 * (y - {prefix}.center) / {prefix.scale}

    Outliers are typically z > 3.5.

    See: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

    :param prefix:
    :param values:
    :return: Returns a dictionary with keys {
                 {prefix}.center - the median of the values
                 {prefix}.scale  - the median absolute deviation of the values
             }
    """
    center = median(values)
    scale = median_absolute_deviation(values)
    return {
        f"{prefix}.center": center,
        f"{prefix}.scale": scale,
        f"{prefix}.threshold": (cutoff * scale)/0.6745
    }
# endregion


# region: agreement metrics
def _nominal_agreement(a: T, b: T) -> int:
    return int(a == b)

def _span_agreement(a: List[Span], b: List[Span]) -> float:
    """
    We define agreement as Jaccard similarity of the spans
    """
    spans: List[WeightedSpan] = WeightedSpan.collapse_spans([
        WeightedSpan(*span) for span in (a + b)])

    if not spans:
        return 1

    union = sum(span.end - span.begin for span in spans)
    # Span count can only be greater than 1 if both a and b match.
    intersection = sum(span.end - span.begin for span in spans if span.weight > 1)

    return intersection / union


def _ordinal_agreement(a: float, b: float, n_values: int) -> float:
    return 1 - abs(a - b) / (n_values - 1)


def _nominal_metric(a: T, b: T) -> int:
    return 1 - _nominal_agreement(a, b)


def _span_metric(a: List[Span], b: List[Span]) -> float:
    """
    We define the metric as the Jaccard distance of the spans
    """
    return 1 - _span_agreement(a, b)


def _interval_metric(a: float, b: float) -> float:
    return (a - b) ** 2


def _ratio_metric(a: float, b: float) -> float:
    return ((a - b) / (a + b)) ** 2


def _ordinal_metric(scores: List[float], a: int, b: int) -> float:
    a, b = min(a, b), max(a, b)
    return (sum(scores[a:b + 1]) - (scores[a] + scores[b]) / 2) ** 2


def krippendorf_alpha(task_worker_value: Dict[T, Dict[W, Any]],
                      metric: str = "nominal",
                      infer_values: bool = True,
                      n_values: Optional[int] = None) -> float:
    """
    Computes Krippendorf's alpha on values.
    :param task_worker_value: is a map from tasks -> workers -> value.
    :param metric: the metric to use between values
    :param infer_values: infer values (useful when task_worker_value is a string value, e.g).
                         If False, task_worker_value must be numeric.
    :param n_values: the maximum number of values. This routine assumes values are contained in [0, n_values).
    :return: the Krippendorf alpha for values.
    """
    # Infer values
    if infer_values:
        values = {value for worker_value in task_worker_value.values() for value in worker_value.values()}
        value_map = {value: i for i, value in enumerate(values)}
        n_values = len(values)
        task_worker_value = {task_id: {worker_id: value_map[value] for worker_id, value in worker_value.items()}
                             for task_id, worker_value in task_worker_value.items()}
    else:
        assert n_values is not None, "Must provide n_values if infer_values is True"

    O = np.zeros((n_values, n_values))
    for _, worker_value in task_worker_value.items():
        # just a single worker did this task, so we can't compute any agreement here.
        if len(worker_value) <= 1:
            continue
        o = np.zeros((n_values, n_values))
        for w, v in worker_value.items():
            for w_, v_ in worker_value.items():
                # Making sure we aren't looking at the same worker.
                if w == w_:
                    continue

                # Update our "confusion matrix" of values.
                o[v, v_] += 1
        n_workers = len(worker_value)
        O += o / (n_workers - 1)

    # Number of guessed values.
    N_v = O.sum(0)
    E = (np.outer(N_v, N_v) - N_v * np.eye(n_values)) / (sum(N_v) - 1)

    if metric == "nominal":
        metric_fn = _nominal_metric
    elif metric == "interval":
        metric_fn = lambda a, b: _interval_metric(a / n_values, b / n_values)
    elif metric == "ratio":
        metric_fn = _ratio_metric
    elif metric == "ordinal":
        metric_fn = lambda a, b: _ordinal_metric(N_v, a, b)
    else:
        raise ValueError(f"Invalid metric {metric}")

    # Get a matrix of "closeness" between the different values.
    delta = np.array([[metric_fn(v, v_) for v in range(n_values)] for v_ in range(n_values)])
    D_o = (O * delta).sum()
    D_e = (E * delta).sum()

    return float(1 - D_o / D_e)


def test_krippendorf_alpha():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = invert_dict({
        'A': {6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, },
        'B': {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, },
        'C': {3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3, },
    })
    assert np.allclose(krippendorf_alpha(data, metric="nominal", infer_values=False, n_values=4), 0.691, atol=5e-3)
    assert np.allclose(krippendorf_alpha(data, metric="interval", infer_values=False, n_values=4), 0.811, atol=5e-3)
    assert np.allclose(krippendorf_alpha(data, metric="ordinal", infer_values=False, n_values=4), 0.807, atol=5e-3)


def pearson_rho(task_worker_values: Dict[T, Dict[W, float]]) -> Dict[W, Optional[float]]:
    """
    Computes pearson rho between each worker's values and the mean value.
    Only makes sense when the values are ordinal.
    :param task_worker_values:
    :return:
    """
    # get task means.
    per_worker_rho = defaultdict(list)
    for _, worker_values in task_worker_values.items():
        if len(worker_values) < 2:
            continue

        avg = mean(list(worker_values.values()))
        for worker, value in worker_values.items():
            per_worker_rho[worker].append([value, avg])

    ret = {}
    for worker, pairs in per_worker_rho.items():
        if len(pairs) > 1:
            pairs_ = np.array(pairs)
            ret[worker] = float(scstats.pearsonr(pairs_.T[0], pairs_.T[1])[0])
        else:
            ret[worker] = None
    return ret


def test_pearson_rho():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = invert_dict({
        'A': {6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, },
        'B': {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, },
        'C': {3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3, },
    })
    rhos = pearson_rho(data)
    assert np.allclose(rhos['A'], 0.947, atol=5e-3)
    assert np.allclose(rhos['B'], 0.909, atol=5e-3)
    assert np.allclose(rhos['C'], 0.984, atol=5e-3)


def pairwise_agreement(task_worker_values: Dict[T, Dict[W, Any]],
                     mode: str = "nominal", n_values: Optional[int] = None) -> float:
    """
    Computes simple agreement on @task_worker_values
    :param task_worker_values:
    :param mode:
    :param infer_values:
    :param n_values:
    :return:
    """
    # Infer values
    if mode == "nominal":
        fn = _nominal_agreement
    elif mode == "span":
        fn = _span_agreement
    elif mode == "ordinal":
        assert n_values is not None, "Must provide n_values if mode is 'ordinal'"
        fn = lambda a, b: _ordinal_agreement(a, b, n_values)
    else:
        raise ValueError("Invalid mode {}".format(mode))

    # rolling average
    ret, n = 0., 0
    for _, worker_values in task_worker_values.items():
        # No agreement with just a single value
        if len(worker_values) < 2:
            continue

        values = sorted(worker_values.values())
        # get unique pairs
        for j, v in enumerate(values):
            for _, v_ in enumerate(values[j + 1:]):
                # compute probability
                ret += (fn(v, v_) - ret) / (n + 1)
                n += 1
    return ret


def test_pairwise_agreement_nominal():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = invert_dict({
        'A': {6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, },
        'B': {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, },
        'C': {3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3, },
    })
    agreement = pairwise_agreement(data, "nominal")
    assert np.allclose(agreement, 0.75, atol=5e-3)


def test_pairwise_agreement_ordinal():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = invert_dict({
        'A': {6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, },
        'B': {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, },
        'C': {3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3, },
    })
    agreement = pairwise_agreement(data, "ordinal", n_values=4)
    assert np.allclose(agreement, 0.895, atol=5e-3)


# Alias for backward compatibility.
simple_agreement = pairwise_agreement


def mean_agreement(task_worker_values: Dict[T, Dict[W, Any]], agg: Dict[T, Any],
                   mode: str = "nominal", n_values: Optional[int] = None) -> float:
    """
    Computes simple agreement on @task_worker_values
    :param task_worker_values:
    :param agg: aggregated values
    :param mode: can be nominal or ordinal. nominal checks for exact equality, ordinal checks for mean.
    :param n_values:
    :return:
    """
    # Infer values
    if mode == "nominal":
        fn = _nominal_agreement
    elif mode == "span":
        fn = _span_agreement
    elif mode == "ordinal":
        assert n_values is not None, "Must provide n_values if mode is 'ordinal'"
        fn = lambda a, b: _ordinal_agreement(a, b, n_values)
    else:
        raise ValueError("Invalid mode {}".format(mode))

    # rolling average
    ret, n = 0., 0
    for task, worker_values in task_worker_values.items():
        # No agreement with just a single value
        if len(worker_values) < 2:
            continue

        agg_value = agg[task]
        for worker, value in worker_values.items():
            # compute probability
            ret += (fn(agg_value, value) - ret)/(n+1)
            n += 1
    return ret


def test_mean_agreement_nominal():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = invert_dict({
        'A': {6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, },
        'B': {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, },
        'C': {3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3, },
    })
    agg = {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2}
    agreement = mean_agreement(data, agg, "nominal")
    assert np.allclose(agreement, 0.884, atol=5e-3)


def test_mean_agreement_ordinal():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = invert_dict({
        'A': {6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, },
        'B': {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, },
        'C': {3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3, },
    })
    agg = {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2}
    agreement = mean_agreement(data, agg, "ordinal", n_values=4)
    assert np.allclose(agreement, 0.948, atol=5e-3)


def mean_agreement_per_worker(task_worker_values: Dict[T, Dict[W, Any]], agg: Dict[T, Any],
                   mode: str = "nominal", n_values: Optional[int] = None) -> Dict[W, float]:
    """
    Computes simple agreement on @task_worker_values
    :param task_worker_values:
    :param agg: aggregated values
    :param mode: can be nominal or ordinal. nominal checks for exact equality, ordinal checks for mean.
    :param n_values:
    :return:
    """
    # Infer values
    if mode == "nominal":
        fn = _nominal_agreement
    elif mode == "span":
        fn = _span_agreement
    elif mode == "ordinal":
        assert n_values is not None, "Must provide n_values if mode is 'ordinal'"
        fn = lambda a, b: _ordinal_agreement(a, b, n_values)
    else:
        raise ValueError("Invalid mode {}".format(mode))

    # rolling average
    ret = defaultdict(list)
    for task, worker_values in task_worker_values.items():
        # No agreement with just a single value
        if len(worker_values) < 2:
            continue

        agg_value = agg[task]
        for worker, value in worker_values.items():
            # compute probability
            ret[worker].append(fn(agg_value, value))
    return {worker: float(np.mean(values)) for worker, values in ret.items()}


def test_mean_agreement_per_worker_nominal():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = invert_dict({
        'A': {6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, },
        'B': {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, },
        'C': {3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3, },
    })
    agg = {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2}
    agreement = mean_agreement_per_worker(data, agg, "nominal")
    assert np.allclose(agreement["A"], 0.888, atol=5e-3)
    assert np.allclose(agreement["B"], 1.0, atol=5e-3)
    assert np.allclose(agreement["C"], 0.818, atol=5e-3)


def test_mean_agreement_per_worker_ordinal():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = invert_dict({
        'A': {6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, },
        'B': {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, },
        'C': {3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3, },
    })
    agg = {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2}
    agreement = mean_agreement_per_worker(data, agg, "ordinal", n_values=4)
    assert np.allclose(agreement["A"], 0.925, atol=5e-3)
    assert np.allclose(agreement["B"], 1.0, atol=5e-3)
    assert np.allclose(agreement["C"], 0.939, atol=5e-3)


def per_worker(task_worker_values: Dict[T, Dict[W, Any]]) -> Dict[W, List[Any]]:
    """
    Transform a task worker matrix into a mapping from workers to list of values.

    :param task_worker_values:
    :return:
    """
    ret = defaultdict(list)
    for worker_values in task_worker_values.values():
        for worker, value in worker_values.items():
            ret[worker].append(value)
    return ret


def micro(fn: Callable[[List[Any]], float], values: Dict[T, List[float]]) -> float:
    return fn(list(chain.from_iterable(values.values())))


def macro(fn: Callable[[List[Any]], float], values: Dict[T, List[float]]) -> Dict[T, float]:
    return {k: fn(vs) for k, vs in values.items()}


# def _factorize(data):
#     """
#     Try to learn turker and task scores as a linear model.
#     """
#     workers = sorted(data.keys())
#     tasks = sorted({hit for hits in data.values() for hit in hits})
#     n_entries = sum(len(hits) for hits in data.values())
#
#     X = np.zeros((n_entries, len(workers) + len(tasks)))
#     Y = np.zeros(n_entries)
#     i = 0
#     for worker, hits in data.items():
#         for task, value in hits.items():
#             X[i, workers.index(worker)] = 1
#             X[i, len(workers) + tasks.index(task)] = 1
#             Y[i] = value
#             i += 1
#
#     return X, Y

# def compute_task_worker_interactions(data, alpha=0.1):
#     """
#     Using a mixed-effects model: y = Wx + Za jk
#     """
#     from sklearn.linear_model import Ridge # typing: ignore
# 
#     data = flatten_dict(data)
#     keys = sorted(data.keys())
#     workers, hits = zip(*keys)
#     workers, hits = sorted(set(workers)), sorted(set(hits))
# 
#     Y = np.zeros(len(keys) + 2)
#     X = np.zeros((len(keys) + 2, len(workers) + len(hits))) # + len(keys)-1))
# 
#     wf = [0 for _ in workers]
#     hf = [0 for _ in hits]
#     for i, (worker, hit) in enumerate(keys):
#         Y[i] = data[worker,hit]
#         wi, hi = workers.index(worker), hits.index(hit)
#         X[i, wi] = 1
#         X[i, len(workers) + hi] = 1
#         wf[wi] += 1
#         hf[hi] += 1
#     # constraint: proportional sum of workers = 0
#     Y[len(keys)], X[len(keys), :len(workers)] = 0, wf
#     # constraint: proportional sum of tasks = 0
#     Y[len(keys)+1], X[len(keys)+1, len(workers):] = 0, hf
# 
#     model = Ridge(alpha=alpha)#, fit_intercept=False)
#     model.fit(X, Y)# - Y.mean())
# 
#     mean = model.intercept_
#     worker_coefs = model.coef_[:len(workers)]
#     hit_coefs = model.coef_[len(workers):]
# 
#     residuals = Y - model.predict(X)
# 
#     ret = {
#         "mean": mean,
#         "worker-std": np.std(worker_coefs),
#         "task-std": np.std(hit_coefs),
#         "residual-std": np.std(residuals),
#         }
# 
#     return ret
