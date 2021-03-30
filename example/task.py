from typing import List, cast, Optional
from texpy.quality_control import QualityControlDecision, generate_explanation
from texpy.experiment import TaskHelper
from texpy.aggregators import *
from texpy.metrics import *
from texpy.util import unmark_for_sanitization
from collections import Counter
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class Task(TaskHelper):
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

        values = [bool(r["Answer"]["output"]) for r in raw_responses]
        return {
            "output": majority_vote(values),
        }

    def make_output(self, input_: dict, agg: dict) -> List[dict]:
        """
        Create well-structured output from input and the aggregated response.

        Args:
            input_:  input given to the HIT
            agg:  return value of aggregate_responses

        Returns:
            A list of objects to be saved in the output data file
        """
        value = agg["output"]

        ret = unmark_for_sanitization(deepcopy(input_))
        ret["label"] = value
        return [ret]

    def compute_metrics(self, inputs: List[dict], outputs: List[List[dict]],
                        agg: List[dict]) -> dict:
        """
        Computes metrics on the input. See `texpy.metrics` for lots of useful metrics.

        Args:
            inputs   : A list of inputs for the task.
            outputs  : A list of outputs; each element contains a list of responses from workers.
            agg      : The aggregated output.


        Returns:
            A dictionary containing all the metrics we care about for this task.
        """
        # Saving responses for future help.
        task_ids = [response['_Meta']['HITId']
                for responses in outputs
                for response in responses
                ]
        worker_ids = [response["_Meta"]["WorkerId"]
                for responses in outputs
                for response in responses
                ]
        task_times = [response["_Meta"]["WorkerTime"]
                for responses in outputs
                for response in responses
                ]
        responses = [response["Answer"]["output"]
                for responses in outputs
                for response in responses
                ]

        labels: Dict[str, Dict[str, str]] = as_task_worker_dict(
            (f"{task_id}", f"{worker_id}", response)
            for task_id, worker_id, response in zip(task_ids, worker_ids, responses)
        )

        agg_task_ids = [responses[0]['_Meta']['HITId'] for responses in outputs if responses]
        agg_labels: Dict[str, str] = {f"{task_id}": response
                                      for task_id, response in zip(agg_task_ids, agg)
                                      }

        return {
            "Label.pairwise_agreement": pairwise_agreement(labels),
            "Label.mean_agreement": mean_agreement(labels, agg_labels),
            "Label.alpha": krippendorf_alpha(labels),
            "Worker.n_responses": macro(len, mean_agreement_per_worker(labels, agg_labels)),
            "Worker.mean_agreement": macro(mean, mean_agreement_per_worker(labels, agg_labels)),
            **distribution("TaskTime", task_times),
            **modified_z_score("TaskTime.zscore", task_times),
        }

    # endregion

    # region: quality control
    def check_quality(self, input_: dict, response: dict, agg: dict, metrics: dict) -> List[
        QualityControlDecision]:
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
        worker_id = response["_Meta"]["WorkerId"]
        # When there's just one worker, we really don't know about
        # agreement rates.
        agreement = metrics["Worker.mean_agreement"].get(worker_id, 0.5)

        # If a worker has agreement < 0.5, all their work is suspect.
        if agreement < 0.5:
            return [QualityControlDecision(
                should_approve=False,
                short_reason="Poor agreement",
                reason=f"The answers you provided differed significantly "
                       "from the other responses we recieved and appear "
                       "wrong.")]
        else:
            return []

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

    # endregion

__all__ = ['Task']
