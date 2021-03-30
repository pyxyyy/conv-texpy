"""
Useful commands for the runtime.
"""

import logging
import sys
from datetime import datetime
from typing import List, Optional, cast
from collections import defaultdict

import yaml
from jinja2 import Template
from tqdm import tqdm

from . import botox
from .experiment import ExperimentBatch
from .util import force_user_input, sanitize

logger = logging.getLogger(__name__)

_AMAZON_CUT = 0.2


def get_reward(config: dict) -> float:
    """
    Ensures that the tasks have a reward and estimated time in
    accordance with the configuration.
    """
    # Make sure that props either defines at least an estimated time +
    # rewardrate or a reward.
    assert 'Reward' in config or ('EstimatedTime' in config and 'RewardRate' in config), \
        "Expected to find either a Reward or RewardRate and EstimatedTime in config"

    if 'Reward' in config:
        return config['Reward']
    else:
        return config['EstimatedTime'] * config['RewardRate'] / 3600


def get_bonus(config: dict, reward_rate_discount: float = 0.9) -> float:
    """
    Ensures that the tasks have a bonus and estimated time in
    accordance with the configuration.

    Args:
       config: The task configuration (task.yaml)
       reward_rate_discount: A discount applied to the reward_rate. Set to 1.0
        to use the reward rate.
    """
    # Make sure that props either defines at least an estimated time +
    # rewardrate or a reward.
    if 'Bonus' in config:
        return config['Bonus']
    elif 'BonusEstimatedTime' in config and 'RewardRate' in config:
        return config['BonusEstimatedTime'] * config['RewardRate'] * reward_rate_discount / 3600
    else:
        return 0


def adjust_for_sandbox(config: dict) -> dict:
    """
    Update the config to test on the sandbox.
    We only use a single assignment, and break qualifications.
    """
    config['MaxAssignments'] = 1
    config['Qualifications'] = []
    config['QualificationTypeId'] = None
    config['Sandbox'] = True

    return config


def launch_task(exp: ExperimentBatch,
                use_prod: bool = False,
                max_hits: Optional[int] = None,
                **variables):
    """
    Launches tasks for an experiment.
    Roughly, we:
        (a) render the input for the experiment to HTML. 
        (b) get the estimated cost and confirm launch with the user.
        (c) actually upload the tasks to AMT.


    Args:
       exp: the experiment to run
       use_prod: if true, we use the AMT prod to run our experiment.
       max_hits: if set, the maximum number of hits to upload.
       variables: any additional variables to be used to render the HTML.
    """
    inputs = exp.inputs
    config = exp.config
    conn = botox.get_client(use_prod)

    # 1. Render all the tasks to HTML
    with exp.open('index.html') as f:
        template = Template(f.read())

    # 2. Tweak properties for the sandbox.
    if not use_prod:
        config = adjust_for_sandbox(config)
        # (Just a sensible default)
        if max_hits is None:
            max_hits = 2

    if max_hits != None:
        inputs = inputs[:max_hits]

    htmls = [template.render({'input': sanitize(input_), 'config': config, **variables, })
             for input_ in tqdm(inputs, desc="rendering inputs")]

    # 3. Make sure we're running the right configurations.
    reward = get_reward(config)
    if "EstimatedTime" in config:
        estimated_time = config["EstimatedTime"]
        reward_rate = 3600 * reward / estimated_time
    else:
        logger.warning("You haven't specified an EstimatedTime in the task configuration.")
        estimated_time = 0
        reward_rate = 0
    total_cost = reward * config['MaxAssignments'] * len(inputs) * (1 + _AMAZON_CUT)

    logger.info(f"We're going to launch {len(inputs)}x{config['MaxAssignments']} tasks {'(really!)' if use_prod else '(in a sandbox)'}")
    logger.info("Using the following qualifications:\n" + yaml.safe_dump(config['Qualifications']))
    logger.info(f"reward=${reward:0.2f}; time={estimated_time}s; rate=${reward_rate:0.2f}")
    logger.info(f"cost=${total_cost:0.2f}; balance=${float(botox.get_account_balance(conn)):0.2f}")

    if force_user_input("Are you sure you want to launch? ", ["y", "n"]) == "n":
        sys.exit(1)

    config["Reward"] = get_reward(config)
    config["Bonus"] = get_bonus(config)
    # Get a batch id for ourselves
    hit_type_id = botox.setup_hit_type(conn, config)
    if use_prod:
        exp.store("task.yaml", config)

    # Actually create our tasks!
    hits, outputs = [], []
    try:
        for html in tqdm(htmls, desc="Launching hits"):
            hits.append(botox.launch_hit(conn, hit_type_id, config, html))
            outputs.append([])  # This will be populated as the task proceeds.
    finally:
        exp.storel('hits.jsonl', hits)
        exp.storel('outputs.jsonl', outputs)


def sync_task(exp: ExperimentBatch, use_prod: bool = False, force_update: bool = False):
    """
    Syncs task for an experiment.
    Roughly, we:
        (a) scan through all the HITs
        (b) get the estimated cost and confirm launch with the user.
        (c) actually upload the tasks to AMT.

    @param exp: the experiment to run
    @param use_prod: if true, we use the AMT prod to run our experiment.
    @param force_update: if true, force an update of each HIT data
    @returns: True iff we made _some_ update to the HIT or output. 
    """
    hits = exp.loadl("hits.jsonl")
    outputs = exp.loadl("outputs.jsonl")
    assert len(outputs) == len(hits), "Uh oh! We should have as many outputs as we have HITs!"

    conn = botox.get_client(use_prod)

    outputs_updated = False
    for hit, output in tqdm(zip(hits, outputs), total=len(hits), desc="syncing hits"):
        # hit and output are modified inplace.
        if botox.sync_hit(conn, hit, output, force_update):
            outputs_updated = True
            exp.storel("hits.jsonl", hits)
            exp.storel("outputs.jsonl", outputs)
    return outputs_updated


def _with_meta(raw: dict, processed: dict) -> dict:
    if "_Meta" in raw:
        if "_Meta" not in processed:
            processed["_Meta"] = {}
        processed["_Meta"].update(raw["_Meta"])
    return processed


def aggregate_task(exp: ExperimentBatch) -> List[dict]:
    inputs = exp.loadl('inputs.jsonl')
    outputs = exp.loadl("outputs.jsonl")

    ret = exp.helper.aggregate_all_responses(inputs, outputs)

    exp.storel("agg.jsonl", ret)
    return ret


def export_task(exp: ExperimentBatch) -> List[dict]:
    """
    Exports data from the experiment.
    :param exp:
    :return:
    """
    inputs = exp.loadl('inputs.jsonl')
    agg = exp.loadl('agg.jsonl')

    ret = []
    for inp, out in tqdm(zip(inputs, agg), total=len(inputs)):
        for obj in exp.helper.make_output(inp, out):
            ret.append(obj)
    exp.storel('data.jsonl', ret)
    return ret


def compute_metrics(exp: ExperimentBatch) -> dict:
    inputs = exp.loadl('inputs.jsonl')
    outputs = cast(List[List[dict]], exp.loadl('outputs.jsonl'))
    agg = exp.loadl("agg.jsonl")

    ret = exp.helper.compute_metrics(inputs, outputs, agg)
    exp.store("metrics.yaml", ret)
    return ret


def check_task(exp: ExperimentBatch):
    """
    Does quality checking for task by calling tasks helper check_quality.
    At the end of this routine, every task response will have its _Meta field updated to include:
    * response['_Meta']['QualityControlDecisions']: a list of quality control decisions (reject or update qual scores)
    * response['_Meta']['ShouldApprove']: The decision whether or not to approve is summarized based on above decisions.
                                          If any decision.should_approve is false, then we store False here.
    * response['_Meta']['QualificationUpdate']: the total update to the workers qualification.
    * response['_Meta']['Bonus']: the total bonus to be paid to the worker.


    TODO: Are these necessary even?
    Additionally, we initialize the following to fields to be false.
    * response['_Meta']['QualificationUpdated']: True when we've updated the qualification.
    * response['_Meta']['BonusPaid']: True when we've paid the bonus.
    @param exp: the experiment
    """
    inputs = exp.loadl('inputs.jsonl')
    outputs = exp.loadl('outputs.jsonl')

    # Aggregate responses.
    aggs = exp.loadl("agg.jsonl")
    # Compute metrics.
    metrics = exp.load("metrics.yaml")

    # Begin checking data.
    try:
        for input_, responses, agg in tqdm(zip(inputs, outputs, aggs), total=len(inputs)):
            for response in responses:
                meta = response["_Meta"]
                # Don't bother checking something that's already been paid for.
                if meta["AssignmentStatus"] in ["Approved", "Rejected"]:
                    meta["ShouldApprove"] = meta["AssignmentStatus"] == "Approved"
                    continue

                decisions = exp.helper.check_quality(input_, response, agg, metrics)
                meta["QualityControlDecisions"] = decisions
                meta["ShouldApprove"] = not decisions or all(decision.should_approve
                        for decision in decisions)
                meta["QualificationUpdated"] = None

                meta["Bonus"] = exp.helper.bonus(input_, response)
                meta["BonusPaid"] = False
    finally:
        exp.storel("outputs.jsonl", outputs)


def pay_task(exp: ExperimentBatch, use_prod: bool = False):
    """
    Pay all the hits in this task.

    @param exp: the experiment to run
    @param use_prod: if true, we use the AMT prod to run our experiment.
    """
    config = exp.config
    hits = exp.loadl('hits.jsonl')
    outputs = exp.loadl('outputs.jsonl')
    metrics = exp.load("metrics.yaml")

    # Get qualification updates aggregated by worker
    qualification_updates: Dict[str, List[QualityControlDecisions]] = defaultdict(list)
    for responses in outputs:
        for response in responses:
            meta = response["_Meta"]
            if "ShouldApprove" not in meta or meta["AssignmentStatus"] != "Submitted":
                continue
            qualification_updates[meta["WorkerId"]].extend(meta["QualityControlDecisions"])

    conn = botox.get_client(use_prod)

    for hit, responses in tqdm(zip(hits, outputs), desc="Updating payments"):
        for response in responses:
            # Note: response and hit are updated in-place here.
            if update_payments(exp, conn, hit, response,
                               redo_rejected_hits=config.get("RedoRejectedHits", False),
                               with_confirmation=False):
                exp.storel("hits.jsonl", hits)
                exp.storel("outputs.jsonl", outputs)
            if pay_bonuses(conn, response):
                exp.storel("outputs.jsonl", outputs)

    if not use_prod:
        logger.info("Not updating qualifications on sandbox")
        return

    qual_id = config.get("QualificationTypeId")
    for worker_id, decisions in tqdm(qualification_updates.items(), desc="Updating qualifications"):
        score = botox.get_qualification(conn, qual_id, worker_id)
        updated_score = exp.helper.update_worker_qualification(worker_id, metrics, decisions, score)
        if updated_score is not None and updated_score != score:
            botox.set_qualification(conn, qual_id, worker_id, updated_score)


def pay_bonuses(conn, response: dict) -> bool:
    """
    Pay bonuses for a single worker.
    Uses response['_Meta']['Bonus'].
    """
    meta = response["_Meta"]
    bonus = float(meta.get("Bonus", 0.))
    if not meta.get("BonusPaid", False) and bonus > 0:
        logger.info(f"Paying {meta['WorkerId']} a bonus of {meta['Bonus']}")
        botox.pay_bonus(conn, meta["WorkerId"], meta["AssignmentId"],
                        amount=meta["Bonus"],
                        reason="Thank you.")
        meta["BonusPaid"] = True
        return True
    else:
        return False


def update_quals(conn, props, response):
    """
    Update qualifications for a worker.
    :param conn:
    :param props: The configuration file to use. Looks for 'QualificationTypeId'
    :param response: The response
    :return:
    """
    meta = response['_Meta']
    if "ShouldApprove" not in meta:
        logger.error("Skipping %s: You must have checked the response to update its qualification",
                     meta["AssignmentId"])
        return False
    if "QualificationUpdated" in meta and meta["QualificationUpdated"]:
        return False

    qual_id = props.get("QualificationTypeId")
    decisions = props.get("QualityControlDecisions")

    # Set QualificationValue to something
    if any(decision.qualification_value is not None for decision in decisions):
        value = max(decision.qualification_value for decision in decisions)
        botox.set_qualification(conn, qual_id, meta["WorkerId"], value,
                                reason=response.get("UpdateQualificationReason"))
    elif any(decision.qualification_update is not None for decision in decisions):
        update = sum(decision.qualification_update for decision in decisions)
        botox.update_qualification(conn, qual_id, meta["WorkerId"], qual_update,
                                    reason=response.get("UpdateQualificationReason"))
    response["QualificationUpdated"] = botox.DATETIME_FORMAT.format(datetime.now())


def update_payments(exp: ExperimentBatch, conn, hit: dict, response: dict,
                    redo_rejected_hits=True, with_confirmation=True) -> bool:
    """
    Updates payments for a HIT.
    :param exp:
    :param conn:
    :param hit:
    :param response:
    :param redo_rejected_hits:
    :param with_confirmation:
    :return: True iff we made some changes to the hit or response.
    """
    meta = response["_Meta"]
    # All the no-update states.
    if "ShouldApprove" not in meta:
        if meta["AssignmentStatus"] == "Submitted":
            logger.warning(f"Skipping response {meta['AssignmentId']} because it has no ShouldApprove field. "
                           f"Run 'texpy check'")
        else:
            logger.error(f"Response {meta['AssignmentId']} has AssignmentStatus={meta['AssignmentStatus']} but "
                         f"has no ShouldApprove field. The response is inconsistent.")
        return False
    elif (meta["AssignmentStatus"] == "Approved" and meta["ShouldApprove"]) or (
            meta["AssignmentStatus"] == "Rejected" and not meta["ShouldApprove"]):
        # All is well.
        return False
    elif meta["AssignmentStatus"] == "Approved" and not meta["ShouldApprove"]:
        logger.warning(f"Response {meta['AssignmentId']} was already approved but ShouldApprove=False. "
                       f"Cannot retroactively reject assignment. Skipping.")
        return False

    if meta["ShouldApprove"]:
        if meta["AssignmentStatus"] == "Rejected":
            logger.warning(f"Response {meta['AssignmentId']} was previously rejected but ShouldApprove=True. "
                           f"Reversing previous decision.")
            # NOTE: If someone was rejected, we simply undo their qual
            # update, not give them a bonus.
            hit['NumberOfAssignmentsRejected'] -= 1

        botox.approve_assignment(conn, response)
        hit['NumberOfAssignmentsApproved'] += 1
    else:
        feedback = exp.helper.rejection_email(response)
        # Get user confirmation.
        print(f"You are about to reject {meta['AssignmentId']} for the following reason:\n{feedback}")

        if with_confirmation:
            confirmation = force_user_input("Please confirm [y/n]: ", ["y", "n"])
        else:
            confirmation = "y"
        if confirmation == "y":
            logger.info("Rejecting assignment %s", meta["AssignmentId"])
            botox.reject_assignment(conn, response, feedback)
            hit['NumberOfAssignmentsRejected'] += 1

            if redo_rejected_hits:
                botox.redo_hit(conn, hit["HITId"], 1)
                hit['MaxAssignments'] += 1
        else:
            logger.warning(f"Skipping rejection of assignment {meta['AssignmentId']}")

    return True
