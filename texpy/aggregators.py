"""
Data aggregation routines.

This package contains a variety of routines that can be used to combine
annotations from multiple workers. These routines will typically be used
by :func:`texpy.experiment.TaskHelper.aggregate_responses`.
"""
from typing import List, TypeVar, Callable, Optional, Tuple, NamedTuple, Dict, Any
from collections import Counter
import numpy as np
import scipy.stats as scstats
import logging

from .util import Span, WeightedSpan, collapse_spans, invert_dict, merge_adjacent_spans


logger = logging.getLogger(__name__)


T = TypeVar("T")
U = TypeVar("U")
W = TypeVar("W")


def majority_interval(lst: List[List[Span]]) -> List[Span]:
    """
    Aggregates a list of intervals

    Example (annotations are the underlines)::
        # Input:
        This is an example
        ----
        ----    ----------
                ----------
        ------------------

        # Output:
        This is an example
        ----    ----------
    """
    # 1. Collapse spans into non-overlapping sets weighted by frequency
    canonical_spans: List[WeightedSpan] = WeightedSpan.collapse_spans([
        WeightedSpan(*span) for spans in lst for span in spans])

    # 3. Filter to only majoritarian spans
    ret = [(span.begin, span.end) for span in canonical_spans if span.weight >= len(lst) / 2]

    return merge_adjacent_spans(ret)


def test_majority_interval():
    # This is an example
    # ----
    # ----    ----------
    #         ----------
    # ------------------
    assert [(0, 4), (8, 18)] == majority_interval([
        [(0, 4), ],
        [(0, 4), (8, 18)],
        [(8, 18), ],
        [(0, 18)],
    ])

    assert [(0, 4), (8, 18)] == majority_interval([
        [(0, 4), ],
        [(8, 18), ],
        [(0, 4), (8, 18)],
        [(0, 18)],
    ])

    # This is an example
    # ----
    # ----    ----------
    #         ----------
    # --------------    
    #   -------------   
    assert [(0, 4), (8, 16)] == majority_interval([
        [(0, 4), ],
        [(0, 4), (8, 18)],
        [(8, 18), ],
        [(0, 15)],
        [(2, 16)],
    ])


def transpose(lst: List[List[T]]) -> List[List[T]]:
    n = len(lst[0])
    assert all(len(lst_) == n for lst_ in lst), "Expected inputs to have uniform lengths"

    return list(zip(*lst))


def apply_list(fn: Callable[[List[T]], U], lst: List[List[T]]) -> List[U]:
    """
    Applies a single function to every element of a list
    """
    if not lst:
        return []
    return [fn(lst_) for lst_ in transpose(lst)]


def majority_vote(lst: List[T]) -> T:
    """
    Returns the most common value
    """
    (elem, _), = Counter(lst).most_common(1)
    return elem


def _normalize(arr: np.array, axis: int = -1) -> np.array:
    """
    Normalize a n-d array over axis.

    Args:
        arr: An array with dimensions $(m_1, ..., m_n)$
        axis: The dimension to average along.
    Returns:
        An array with dimensions $(m_1, ..., m_n)$ that sums to 1 when summing over `axis`
    """
    return arr / arr.sum(axis, keepdims=True)


def dawid_skene_model(
        task_worker_values: Dict[T, Dict[W, U]],
        alpha: float = 1e-5,
        beta: float = 1e-2,
        eps: float = 0.1,
        max_iters: int = 100) -> Tuple[Dict[T, U], Dict[W, float]]:
    r"""
    Aggregates worker responses using a Dawid-Skene model.

    The Dawid-Skene is a simple probablistic graphical model. Here is the generative process:

    * We draw a multinomial distribution over labels $\pi ~ Dirichlet(\alpha)$.
      In practice, we assume $\alpha$ is given and is a uniform distribution. This is just Laplace
      smoothing
    * We draw multinomial confusion matrices for each annotator $a$: $C^a_{k} ~ Dirichlet(\beta)$,
      where $k \in [K]$ the set of true values.
    * For each task $t$, we draw it's true label $r^*_t ~ \pi$.
    * For each task $t$ and annotator (that performed the task), we draw the observed response:
      $r^a_t ~ C^a_{r^*_t}$.

    The results of the model are inferred using the EM algorithm.

    Args:
        task_worker_values: A mapping from tasks to workers and their responses.
        alpha: Additive smoothing to be applied to the label distribution (typically rather small)
        beta: Additive smoothing to be applied to worker confusion matrices (typically a bit larger
              than alpha, though this shouldn't be too large if there are workers who have worked
              on very few tasks).
        eps: Epsilon used to determine convergence on the task label distribution. This can
             typically be fairly large -- 0.1 is not unreasonble as a value if you have ~1000 tasks.
        max_iters: The maximum iterations to run the EM algorithm for.

    Returns:
        A tuple of:
            - the most likely value (U) for a given task
            - per-worker accuracies
    """
    assert max_iters > 0

    # Encode tasks, workers and values categorically.
    tasks = sorted(task for task in task_worker_values)
    task_to_idx = {task: i for i, task in enumerate(tasks)}
    workers = sorted({worker for responses in task_worker_values.values() for worker in responses})
    worker_to_idx = {worker: i for i, worker in enumerate(workers)}
    values = sorted({value for responses in task_worker_values.values()
                     for value in responses.values()})
    value_to_idx = {value: i for i, value in enumerate(values)}

    # We initialize task responses using raw occurrence counts
    task_labels = np.zeros((len(tasks), len(values)))
    for task, responses in task_worker_values.items():
        for _, value in responses.items():
            task_labels[task_to_idx[task], value_to_idx[value]] += 1
    # normalize per task
    task_labels = _normalize(task_labels + alpha)

    for i in range(max_iters):
        # M-step
        # Set label distribution and worker confusion based on our current task labels
        worker_confusion = np.zeros((len(workers), len(values), len(values)))
        for task, responses in task_worker_values.items():
            for worker, value in responses.items():
                worker_confusion[worker_to_idx[worker], :, value_to_idx[value]] += task_labels[
                    task_to_idx[task]]
        worker_confusion = _normalize(worker_confusion + beta)
        label_distribution = _normalize(task_labels.sum(0))

        # E-step
        task_labels_ = np.zeros((len(tasks), len(values)))
        for task, responses in task_worker_values.items():
            log_task_distribution = np.zeros(len(values))
            for worker, value in responses.items():
                log_task_distribution += np.log(
                    worker_confusion[worker_to_idx[worker], :, value_to_idx[value]])
            log_task_distribution += np.log(label_distribution)
            task_labels_[task_to_idx[task]] = np.exp(
                log_task_distribution - np.logaddexp.reduce(log_task_distribution))
        task_labels_ = _normalize(task_labels_ + alpha)

        # Compute convergence distance
        update_norm = np.linalg.norm(task_labels - task_labels_)
        task_labels = task_labels_

        with np.printoptions(precision=3, suppress=True):
            logger.info(
                f"Epoch {i}: update={np.array([update_norm])}, label_dist={label_distribution}")

        if update_norm < eps:
            break

    # Convert our matrices back into object form
    ret_task = {task: values[value_idx]
                for task, value_idx in zip(tasks, task_labels.argmax(1).tolist())}
    ret_workers = {worker: (np.diag(confusion).sum()/confusion.sum()).tolist()
                   for worker, confusion in zip(workers, worker_confusion)}
    return ret_task, ret_workers


def test_dawid_skene():
    data = invert_dict({
        'A': {6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, },
        'B': {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, },
        'C': {3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3, },
    })
    task_values, worker_scores = dawid_skene_model(data)
    assert task_values == {
        1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2}
    assert np.allclose(worker_scores["A"], 0.95, atol=1e-2)
    assert np.allclose(worker_scores["B"], 0.90, atol=1e-2)
    assert np.allclose(worker_scores["C"], 0.88, atol=1e-2)


try:
    from sklearn.linear_model import Ridge

    class LabelSmoothedLogisticRegression(Ridge):
        def __init__(self, n_classes: int = 2, label_smoothing_coef: float = 0.01, alpha: float = 1e0):
            super().__init__(alpha=alpha)
            self.label_smoothing_coef = label_smoothing_coef
            self.n_classes = n_classes

        def fit(self, X, y, sample_weight=None):
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            # unravel the classes
            n_samples, = y.shape

            y_ = np.zeros((n_samples, self.n_classes))
            y_.fill(self.label_smoothing_coef / (self.n_classes - 1))
            y_[list(range(n_samples)), y] = 1 - self.label_smoothing_coef
            y_ = np.log(y_)
            super().fit(X, y_, sample_weight)

        def predict(self, X):
            y_ = super().predict(X)
            return y_.argmax(1)

        def predict_proba(self, X):
            return np.exp(self.predict_log_proba(X))
        
        def predict_log_proba(self, X):
            y_ = super().predict(X)
            y = y_ - np.logaddexp.reduce(y_, 1, keepdims=True)
            return y

        def score(self, X, y):
            n_samples, = y.shape
            y_ = np.zeros((n_samples, self.n_classes))
            y_[list(range(n_samples)), y] = 1
            return self.score(X, y_)

    def test_label_smoothing():
        N, D, K = 1000, 10, 3
        Xk = np.random.randn(K, D)
        y = np.random.randint(0, K, N)
        X = Xk[y, :] + 0.01 * np.random.randn(N, D)

        # With smoothing
        model = LabelSmoothedLogisticRegression(K)
        model.fit(X, y)
        assert np.allclose(model.predict(X), y)
        assert (model.predict_proba(X)[list(range(N)), y] > 0.9).all()

    def dawid_skene_model_with_features(
            task_worker_values: Dict[T, Dict[W, U]],
            features: Dict[T, np.array],
            alpha: float = 1e-5,
            beta: float = 1e-2,
            gamma: float = 1e0,
            eps: float = 0.1,
            max_iters: int = 100) -> Dict[T, U]:
        r"""
        Aggregates worker responses using a featurized Dawid-Skene model.

        See `dawid_skene_model` for more details about the conventional Dawid-Skene model.

        This version applies the following discriminative process:

        * We draw a multinomial distribution over labels $\pi ~ Dirichlet(\alpha)$.
          In practice, we assume $\alpha$ is given and is a uniform distribution---
          this is just Laplace smoothing. Let K be the number of labels.
        * We draw a K x K x D matrix for each annotator $a$: $C^a_{k} ~ Gaussian(I, \beta)$,
          where $k \in [K]$, the set of labels.
        * For each task $t$, we draw it's true label $r^*_t ~ \pi$.
        * For each task $t$ and annotator (that performed the task), we draw the observed response:
          $r^a_t ~ C^a_{r^*_t}$.

        The results of the model are inferred using the EM algorithm.

        Args:
            task_worker_values: A mapping from tasks to workers and their responses.
            features: A mapping from tasks to a vector featurizing it.
            alpha: Additive smoothing to be applied to the label distribution (typically rather small)
            beta: Additive smoothing to be applied to worker confusion matrices (typically a bit larger
                  than alpha, though this shouldn't be too large if there are workers who have worked
                  on very few tasks).
            gamma: L2 regularization coefficient (corresponds to `sklearn.linear_model.LinearRegression.alpha`).
            eps: Epsilon used to determine convergence on the task label distribution. This can
                 typically be fairly large -- 0.1 is not unreasonble as a value if you have ~1000 tasks.
            max_iters: The maximum iterations to run the EM algorithm for.

        Returns:
            The most likely value (U) for a given task under the model
        """
        assert max_iters > 0

        # Encode tasks, workers and values categorically.
        tasks = sorted(task for task in task_worker_values)
        task_to_idx = {task: i for i, task in enumerate(tasks)}
        workers = sorted({worker for responses in task_worker_values.values() for worker in responses})
        worker_to_idx = {worker: i for i, worker in enumerate(workers)}
        values = sorted({value for responses in task_worker_values.values()
                         for value in responses.values()})
        value_to_idx = {value: i for i, value in enumerate(values)}

        # We initialize task responses using raw occurrence counts
        task_labels = np.zeros((len(tasks), len(values)))
        for task, responses in task_worker_values.items():
            for _, value in responses.items():
                task_labels[task_to_idx[task], value_to_idx[value]] += 1
        # normalize per task
        task_labels = _normalize(task_labels + alpha)

        # Summarize the data once and for all
        X = np.array([features[task] for task in tasks])
        ix = {worker: [] for worker in workers}
        Y = {worker: [] for worker in workers}
        for i, task in enumerate(tasks):
            for worker, value in task_worker_values[task].items():
                ix[worker].append(i)
                Y[worker].append(value_to_idx[value])

        # The task vector dimension
        for i in range(max_iters):
            # M-step
            # Set label distribution and worker confusion based on our current task labels
            worker_models = {worker_id: {value: LabelSmoothedLogisticRegression(
                alpha=gamma, n_classes=len(values), label_smoothing_coef=beta)
                        for value in values} for worker_id in workers}
            for worker in workers:
                x, y, w = X[ix[worker]], Y[worker], task_labels[ix[worker]]
                for value in values:
                    worker_models[worker][value].fit(x, y, w[:, value_to_idx[value]])

            label_distribution = _normalize(task_labels.sum(0))

            # E-step
            # Re-estimate task labesl
            task_labels_ = np.zeros(task_labels.shape)
            for task, responses in task_worker_values.items():
                log_task_distribution = np.zeros(len(values))
                for worker, value in responses.items():
                    for k in range(len(values)):
                        log_task_distribution[k] += worker_models[worker][k].predict_log_proba(
                                features[task].reshape(1,-1))[0, value_to_idx[value]]
                log_task_distribution += np.log(label_distribution)
                task_labels_[task_to_idx[task]] = np.exp(
                    log_task_distribution - np.logaddexp.reduce(log_task_distribution))
            task_labels_ = _normalize(task_labels_ + alpha)

            # Compute convergence distance
            update_norm = np.linalg.norm(task_labels - task_labels_)
            task_labels = task_labels_

            with np.printoptions(precision=3, suppress=True):
                logger.info(
                    f"Epoch {i}: update={np.array([update_norm])}, label_dist={label_distribution}")

            if update_norm < eps:
                break

        # Convert our matrices back into object form
        ret_task = {task: values[value_idx]
                    for task, value_idx in zip(tasks, task_labels.argmax(1).tolist())}
        return ret_task


    def test_dawid_skene_with_features():
        data = invert_dict({
            'A': {6: 2, 7: 3, 8: 0, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, },
            'B': {1: 0, 3: 1, 4: 0, 5: 2, 6: 2, 7: 3, 8: 2, },
            'C': {3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3, },
        })
        def one_hot(i: int, d=4) -> np.array:
            ret = np.zeros(d)
            ret[i] = 1
            return ret

        features = {
                1: one_hot(0),
                3: one_hot(1),
                4: one_hot(0),
                5: one_hot(2),
                6: one_hot(2),
                7: one_hot(3),
                8: one_hot(0),
                9: one_hot(1),
                10: one_hot(0),
                11: one_hot(0),
                12: one_hot(2),
                13: one_hot(2),
                15: one_hot(2),
                }

        task_values = dawid_skene_model_with_features(data, features)
        assert task_values == {
            1: 0, 3: 1, 4: 0, 5: 2, 6: 3, 7: 3, 8: 2, 9: 1, 10: 0, 11: 0, 12: 2, 13: 2, 15: 3}

except ImportError:
    class LabelSmoothedLogisticRegression:
        pass

    def dawid_skene_model_with_features(*args):
        logger.fatal("Could not import sklearn (scikit-learn). Please install scikit-learn, "
                     "'pip install scikit-learn'")
        raise NotImplementedError()


def _float(fn: Callable[[T], U]) -> Callable[[T], float]:
    return lambda x, *args, **kwargs: float(fn(x, *args, **kwargs))


def _guard(fn: Callable[[T], U]) -> Callable[[T], Optional[U]]:
    return lambda x, *args, **kwargs: fn(x, *args, **kwargs) if x else None


mean = _guard(_float(np.mean))
std = _guard(_float(np.std))
median = _guard(_float(np.median))
median_absolute_deviation = _guard(_float(scstats.median_absolute_deviation))
mode = _guard(_float(scstats.mode))
trim_mean = _guard(_float(scstats.trim_mean))
percentile = _guard(_float(np.percentile))

__all__ = ['mean', 'std', 'median', 'median_absolute_deviation', 'mode', 'trim_mean', 'percentile',
           'transpose', 'apply_list', 'majority_vote', 'majority_interval', 'dawid_skene_model', 'dawid_skene_model_with_features']
