"""
Experiment scaffolding (texpy.experiment)
=========================================

This module defines the `TaskHelper` base class that users can subclass
to configure how the data for a particular experiment should be
processed.

.. todo::
    Direct readers to a tutorial on how to define their own task helper.


It additionally defines the `Experiment` and `ExperimentBatch` classes
that define the experiment structure: an experiment contains one or more
batches. The root experiment directory defines a task.py and task.yaml
file that will be used by each batch. Each batch will first look for
a task.py or task.yaml in its own directory before using the default in
its parent directory, allowing you to define a specific task
configuration for a particular batch.
"""

import os
import yaml
import logging
from importlib.util import spec_from_file_location, module_from_spec
from typing import TextIO, List, cast, Optional
from .util import load_jsonl, save_jsonl, SimpleObject, first
from .quality_control import QualityControlDecision

logger = logging.getLogger(__name__)


class TaskHelper:
    """
    This entire file defines handles to process a task: when to give
    bonuses, how to parse and aggregate the input and how to check
    quality.
    """

    # region: task specification
    def bonus(self, input_, response) -> int:
        """
        An (optional) task-specific bonus.
        This could be a bonus for e.g. completing a tutorial or based on the
        length of the task.
        """
        return 0
    # endregion

    # region: data validation and aggregation
    def aggregate_all_responses(self, inputs: List[dict],
                                outputs: List[List[dict]]) -> List[dict]:
        """
        Aggregates all responses in the task

        Args:
            inputs: A list of all the raw input provided to this task.
            outputs: A list of task responses. Each raw_response is
                           a dictionary with key-value pairs from the
                           form's fields.
        Returns:
            A list of dictionaries that represent the aggregated responses for each task.
            The responses are epxected to contain the same format as individual elements of outputs.
        """
        return [self.aggregate_responses(input_, responses)
                for input_, responses in zip(inputs, outputs)]

    def aggregate_responses(self, input_: dict, raw_responses: List[dict]) -> dict:
        """
        Aggregates multiple raw responses.
        See texpy.aggregation for many useful aggregation functions.

        Args:
            input_: The raw input provided to this task.
            raw_responses: A list of raw_responses. Each raw_response is
                           a dictionary with key-value pairs from the
                           form's fields.
        Returns:
            A dictionary that aggregates the responses with the same keys.
        """
        pass

    def make_output(self, input_: dict, agg: dict) -> List[dict]:
        """
        Create well-structured output from input and the aggregated response.

        Args:
            input_:  input given to the HIT
            agg:  return value of aggregate_responses

        Returns:
            A list of objects to be saved in the output data file
        """
        pass

    def compute_metrics(self, inputs: List[dict], outputs: List[List[dict]], agg: List[dict]) -> dict:
        """
        Computes metrics on the input. See `texpy.metrics` for lots of useful metrics.

        Args:
            inputs   : A list of inputs for the task.
            outputs  : A list of outputs; each element contains a list of responses from workers.
            agg      : The aggregated output.


        Returns:
            A dictionary containing all the metrics we care about for this task.
        """
        pass
    # endregion

    # region: quality control
    def check_quality(self, input_: dict, response: dict, agg: dict, metrics: dict) -> List[QualityControlDecision]:
        """
        The quality checking routine.

        Args:
            input_  : input object given to the HIT
            response: A worker's response (an element of outputs)
            agg     : Aggregated response for this task (as returned by aggregate_responses())
            metrics : Metrics computed over all the data (as returned by compute_metrics())

        Returns:
            a list of quality control decisions. Each decision weighs on
            accepting or rejecting the task and gives some feedback back
            to the worker. We reject if any of the returned decisions
            are to reject the HIT.
        """
        pass

    def update_worker_qualification(self,
            worker_id: str,
            metrics: dict,
            decisions: List[QualityControlDecision],
            current_qualification_value: int
            ) -> int:
        """
        The quality checking routine.

        Args:
            input_  : input object given to the HIT
            response: A worker's response (an element of outputs)
            agg     : Aggregated response for this task (as returned by aggregate_responses())
            metrics : Metrics computed over all the data (as returned by compute_metrics())

        Returns:
            a list of quality control decisions. Each decision weighs on
            accepting or rejecting the task and gives some feedback back
            to the worker. We reject if any of the returned decisions
            are to reject the HIT.
        """
        if any(decision.qualification_value is not None for decision in decisions):
            return max(decision.qualification_value or 0
                       for decision in decisions)
        elif any(decision.qualification_update is not None for decision in decisions):
            return current_qualification_value + sum(decision.qualification_update or 0
                                                     for decision in decisions)
        else:
            return current_qualification_value

    @property
    def rejection_email_format(self) -> str:
        """
        The text of an email sent out to workers when their work has
        been rejected.

        Note: the maximum length of this list is ? characters.

        If None, do not respond.
        """
        return """\
Hello {response[_Meta][WorkerId]},
  We are unfortunately rejecting your work for HIT {response[_Meta][HITId]}.
We've tried our best to ensure that our rejections are fair and
deserved; here are our reasons:
{reasons}
If you still feel treated unfairly, please contact us."""

    def rejection_email(self, response: dict, char_limit: int = 1024) -> str:
        """
        The text of an email sent out to workers when their work has
        been rejected.

        Note: the maximum length of this list is ? characters.

        If None, do not respond.
        """
        template = self.rejection_email_format
        reasons: List[QualityControlDecision] = response["_Meta"].get("QualityControlDecisions", [])

        ret = template.format(
            response=response,
            reasons="\n".join("- {}".format(reason.reason) for reason in reasons)
        )

        if len(ret) >= char_limit:
            logger.warning(f"Could not report full feedback for {response['_Meta']['AssignmentId']};"
                           f" using parsimonious version.")
            ret = template.format(
                response=response,
                reasons="\n".join("- {}".format(reason.short_reason) for reason in reasons)
            )

        if len(ret) >= char_limit:
            logger.warning("Could not report parsimonious feedback for {response['_Meta']['AssignmentId']};"
                           " using truncated version.")

            backup_version = "... (message truncated): if you would like further explanation " \
                f"about our decision, please contact us."
            ret = ret[:(char_limit - len(backup_version))] + backup_version

        assert len(ret) < char_limit

        return ret
    # endregion


class Experiment(object):
    """
    Captures the root of an Experiment

    Each experiment defines some common configuration using the
    ``task.yaml`` and ``task.py`` files. Most of the time, one interacts
    with a specific *experiment batch*, which is captured in
    `ExperimentBatch`.

    This class defines a number of useful routines that allow users to
    retrieve files relative to the experiment root, as well as to load
    the `TaskHelper` associated with this experiment.
    """

    def __init__(self, root: str):
        """
        Construct an Experiment

        Args:
            root: The path to the root Experiment directory.
        """
        self.root = root

        #: Lazy storage for `self.config`
        self._config: Optional[SimpleObject] = None
        #: Lazy storage for `self.helper`
        self._helper: Optional[TaskHelper] = None

    def __repr__(self):
        return f"<ExpBatch: {self.root}>"

    def __str__(self):
        return self.basename

    # region: io
    def path(self, fname: Optional[str] = None) -> str:
        """
        Gets the path to a file relative to the experiment root.

        Args:
            fname: An optional filename to retrieve the path of. If none
            is provided, we will return the path of the root.
        Returns:
            If `fname` is provided, the absolute path to `fname`, and
            the absolute path to the experiment root otherwise.
        """
        return os.path.join(self.root, fname) if fname else self.root

    @property
    def mypath(self) -> str:
        """
        An alias for `self.path()` that can be used in format strings.
        """
        return self.path()

    @property
    def basename(self) -> str:
        """
        The basename of an Experiment directory. This is useful in
        format strings.
        """
        return os.path.basename(os.path.realpath(self.root))

    def exists(self, path: Optional[str] = None) -> bool:
        """
        Test if a path relative to the experiment root exists.

        Args:
            path: An optional path to test. If no path is provided, we
            will test whether the experiment root exists.

        Returns:
            `True` iff `path` exists in the local filesystem.
        """
        return os.path.exists(self.path(path))

    def ensure_exists(self, path: Optional[str] = None) -> None:
        """
        Ensure that a directory exists by creating it if it doesn't.

        Args:
            path: An optional path to test. If no path is provided, we
            will use the experiment root.
        """
        path = self.path(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def open(self, fname: str, *args, **kwargs) -> TextIO:
        """
        Opens a file relative to the experiment directory

        Args:
            fname: The (relative) path to the file to open.
            *args: additional positional arguments to `open`.
            *kwargs: additional keyword arguments to `open`.
        Returns:
            A file handle to `fname`.
        """
        return open(self.path(fname), *args, **kwargs)

    def load(self, fname: str) -> SimpleObject:
        """
        Loads a configuration file, assumed to be YAML format.

        Args:
            fname: The (relative) path to the file to read.
        Returns:
            A dictionary parsed from `fname`.
        """
        with self.open(fname) as f:
            return yaml.safe_load(f)

    def loadl(self, fname: str) -> List[SimpleObject]:
        """
        Loads a list of objects from a file assumed to be in JSONL
        format.

        Args:
            fname: The (relative) path to the file to read.
        Returns:
            A list of dictionary objects parsed from `fname`.
        """
        with self.open(fname) as f:
            return load_jsonl(f)

    def store(self, fname: str, obj: SimpleObject):
        """
        Save a configuration file in YAML format.

        Args:
            fname: The (relative) path to the file to write to.
            obj: The dictionary to write
        """
        with self.open(fname, "w") as f:
            yaml.safe_dump(obj, f, indent=2, sort_keys=True) # type:ignore

    def storel(self, fname: str, objs: List[SimpleObject]):
        """
        Save a list of objects to a file in JSONL format.

        Args:
            fname: The (relative) path to the file to write to.
            objs: A list of objects to write
        """
        with self.open(fname, "w") as f:
            save_jsonl(f, objs)
    # endregion

    # region: commands
    def new(self) -> 'ExperimentBatch':
        """
        Create a new experiment batch.

        The newly created experiment batch will have a batch index that
        is larger than the existing batches.
        """
        batches = self.batches()
        if batches:
            ret = ExperimentBatch(self.root, max(batch.idx for batch in batches) + 1)
        else:
            ret = ExperimentBatch(self.root, 0)
        self.ensure_exists(str(ret.idx))
        return ret

    def batches(self) -> List['ExperimentBatch']:
        """
        Get all experiment batches.
        """
        ret = []
        for dirname in os.listdir(self.path()):
            # We only care about directories that are numerals
            if not os.path.isdir(self.path(dirname)) or not dirname.isdigit():
                continue
            batch = int(dirname)
            ret.append(ExperimentBatch(self.root, batch))
        return sorted(ret, key=lambda e: e.idx)

    def find(self, idx: Optional[int]) -> 'ExperimentBatch':
        batches = self.batches()
        if not batches:
            raise IndexError("No batches currently exist. Please create a new one using `new`")
        if idx is not None:
            batch = first(batch for batch in batches if batch.idx == idx)
            if batch is None:
                raise IndexError(f"Could not find batch {idx}. Valid choices are: " +
                        ' '.join(str(batch_.idx) for batch_ in batches))
            return batch
        else:
            return batches[-1]
    # endregion

    # region: Getters for common experiment things.
    @property
    def config(self) -> SimpleObject:
        """
        Return the task configuration from `task.yaml`.
        """
        if self._config is None:
            self._config = self.load("task.yaml")
        return self._config

    @property
    def helper(self) -> TaskHelper:
        """
        Returns the task helper module from `task.py`.
        """
        if self._helper is None:
            # This sequence of magic allows us to import a python module.
            spec = spec_from_file_location("texpy.task", self.path("task.py"))
            assert spec is not None and spec.loader is not None
            module = module_from_spec(spec)
            assert module is not None
            spec.loader.exec_module(module)  # type: ignore

            assert hasattr(module, 'Task')
            self._helper = cast(TaskHelper, module.Task())  # type: ignore
        return self._helper
    # endregion


class ExperimentBatch(Experiment):
    """
    An experiment batch extends experiment with batch-specific utilities.

    While `Experiment` defines common utilities across multiple
    Experiment batches, e.g. the TaskHelper (task.py) or configuration
    (task.yaml), the ExperimentBatch exposes files that are specific the
    given batch like its inputs, outputs, etc.
    """

    def __init__(self, root: str, idx: int):
        """
        Constructs an ExperimentBatch

        Args:
            root: The path of the containing Experiment.
            idx: The index of this particular batch.
        """
        super().__init__(root)
        self.idx = idx

        self._inputs: Optional[List[SimpleObject]] = None

    def __repr__(self):
        return f"<ExpBatch: {self.root}/{self.idx}>"

    def __str__(self):
        return f"{self.basename}/{self.idx}"

    # region: io
    def path(self, fname: Optional[str] = None) -> str:
        """
        Gets the path to a file relative to the experiment batch. If
        this file doesn't exist in the current path, we will look for
        the file in the root directory. If this too doesn't exist, we
        will return the file relative to the experiment batch.

        Args:
            fname: An optional filename to retrieve the path of. If none
            is provided, we will return the path of the batch.
        Returns:
            If `fname` is provided, the absolute path to `fname`, and
            the absolute path to the experiment root otherwise.
        """
        local_path = os.path.join(self.root, str(self.idx))
        if fname:
            local_path = os.path.join(local_path, fname)

        if os.path.exists(local_path):
            return local_path

        root_path = super().path(fname)
        if os.path.exists(root_path):
            return root_path

        return local_path
    # endregion

    # region: Getters for common experiment things.
    @property
    def inputs(self) -> List[SimpleObject]:
        """
        Returns the list of inputs from `inputs.jsonl`.
        """
        if self._inputs is None:
            self._inputs = self.loadl("inputs.jsonl")
        return self._inputs
    # endregion


__all__ = ["Experiment", "ExperimentBatch", "TaskHelper"]
