#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turk experiment helper.

This utility helps you manage the lifecycle of crowd-work tasks on
Amazon Mechanical Turk (AMT). Here's a simple diagram of a typical
lifecycle:


  init†
   ↓
  new ---> view ---> launch --> sync --> check --> pay --> export
   |   └- new -u ┘                                  |  
   └----------------- (new batch)-------------------┴----> clear‡


†: You only need to run `init` the first time you start an experiment:
   it will create the appropriate `task.py`, `task.yaml`, as well as a
   `config.yaml` file for custom hooks.
‡: Running clear will nuke the data for your experiment from AWS's
   servers: make sure you either don't want that data or that you have a
   secure backup before running this command.

You can get more help for each command below.
"""
import logging
import os
import shutil
import sys
import time
from pprint import pprint
from subprocess import check_output, CalledProcessError
from typing import List, cast

import yaml
from tqdm import tqdm, trange

from . import botox
from .commands import launch_task, sync_task, check_task, pay_task, aggregate_task, compute_metrics, get_reward, \
    export_task
from .experiment import Experiment
from .server import serve_viewer
from .util import force_user_input, first
from .quality_control import QualityControlDecision

logger = logging.getLogger(__name__)

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _run(cmd: str):
    """
    Runs a command, throwing an error if it fails.
    """
    logger.info(f"Running: {cmd}")
    check_output(cmd, shell=True)


def _run_hook(cmds: List[str], **kwargs):
    """
    Runs a hook specified by a sequence of bash commands.
    """
    for cmd in cmds:
        _run(cmd.format(**kwargs))


def _get_variables(exp:Experiment, config: dict) -> dict:
    """Process the configuration's variables before rendering it"""
    return {key: value.format(exp=exp) for key, value in config.get("variables", {}).items()}


def do_init(args):
    """
    Initializes a new experiment directory with template task.yaml and
    task.py files.
    """
    template_files = ["task.py", "task.yaml"]
    exp = Experiment(args.root)

    for fname in template_files:
        if exp.exists(fname):
            logger.info("Skipping creating task.py because it already "
                    "exists. Please delete or rename it if you would like to "
                    "restore the template version")
        else:
            _run(f"cp {TEMPLATE_DIR}/{fname} {args.root}")
    logger.info("Ready to start running experiments! Run `new` to create a new batch.")


def do_new(args):
    """
    Creates a new experiment batch or updates an existing one.
    """
    if args.update:
        exp = Experiment(args.root).find(args.idx)
        logger.info(f"Updating existing {exp}")
    else:
        assert args.idx is None, "You cannot provide an index unless you wish to update the experiment"
        batches = Experiment(args.root).batches()

        exp = Experiment(args.root).new()
        logger.info(f"Creating a new experiment {exp}")

        if batches:
            batch = batches[-1]
            try:
                _run(f"cp {batch.path('inputs.jsonl')} {exp.path()}")
            except CalledProcessError as e:
                logger.exception("Could not copy file from previous "
                        "directory. We will use the template version "
                        "instead")
                _run(f"cp {TEMPLATE_DIR}/inputs.jsonl {exp.path()}")
        else:
            _run(f"cp {TEMPLATE_DIR}/inputs.jsonl {exp.path()}")

        _run_hook(args.config["hooks"].get("init/pre", []), exp=exp, **_get_variables(exp, args.config))
    _run_hook(args.config["hooks"].get("init/post", []), exp=exp, **_get_variables(exp, args.config))


def do_view(args):
    """
    View an experiment.
    """
    exp = Experiment(args.root).find(args.idx)
    serve_viewer(exp, args.port, **_get_variables(exp, args.config))


def do_launch(args):
    # 0. Find experiment dir.
    exp = Experiment(args.root).find(args.idx)

    # Make sure we aren't overwriting a hits.jsonl because without
    # HITIds we won't be able to pay or clear old tasks.
    if exp.exists('hits.jsonl'):
        logger.fatal(f"You've already launched some HITs! If you are " +
                     f"*sure* that's not the case, run `texpy clear {exp.idx}` " +
                     f"OR delete {exp.path('hits.jsonl')}")
        sys.exit(1)

    # Run pre-launch hooks.
    _run_hook(args.config["hooks"].get("launch/pre", []), exp=exp, **_get_variables(exp, args.config))
    launch_task(exp, use_prod=args.prod, max_hits=args.max_hits, **_get_variables(exp, args.config))
    _run_hook(args.config["hooks"].get("launch/post", []), exp=exp, **_get_variables(exp, args.config))


def _task_summary(exp: Experiment) -> str:
    hits: List[dict] = cast(List[dict], exp.loadl('hits.jsonl'))

    pending = sum(hit['NumberOfAssignmentsPending'] for hit in hits)
    total = sum(hit['MaxAssignments'] for hit in hits)
    received = sum(hit['NumberOfAssignmentsCompleted'] for hit in hits)
    return f"{received / total * 100:d}% done: received of {received} of {total} responses for {len(hits)} HITs, " \
        f"with {pending} pending"


def do_sync(args):
    # 0. Find experiment dir.
    exp = Experiment(args.root).find(args.idx)

    # 1. In a loop, based on input.
    while True:
        had_update = sync_task(exp, use_prod=args.prod, force_update=args.force)

        if args.auto_qc and had_update:
            check_task(exp)
            pay_task(exp, use_prod=not args.prod)

        # get summary
        hits = exp.loadl('hits.jsonl')
        outputs = exp.loadl('outputs.jsonl')

        pending = sum(hit['NumberOfAssignmentsPending'] for hit in hits)
        total = sum(hit['MaxAssignments'] for hit in hits)
        received = sum(1 for responses in outputs for response in responses
                       if response["_Meta"]["AssignmentId"] is not None)
        logger.info(
            f"{received / total * 100:.0f}% done: received of {received} of {total} responses for {len(hits)} HITs, " +
            f"with {pending} pending")

        if not args.blocking or received == total:
            break
        else:
            for _ in trange(int(args.wait_time / 10), desc="Waiting..."):
                time.sleep(10)

    # 2. Update parsed, aggregates and metrics.
    aggregate_task(exp)
    metrics = compute_metrics(exp)
    # Print metrics.
    print("=== Metrics ===")
    yaml.safe_dump(metrics, sys.stdout)


def do_check(args):
    """
    Check processes the input to produce

    :param args:
    :return:
    """
    exp = Experiment(args.root).find(args.idx)

    aggregate_task(exp)
    metrics = compute_metrics(exp)
    # Print metrics.
    print("=== Metrics ===")
    yaml.safe_dump(metrics, sys.stdout)

    check_task(exp)

    outputs = exp.loadl("outputs.jsonl")

    # Now confirm all rejects.
    rejects = [response for responses in outputs for response in responses if not response["_Meta"]["ShouldApprove"]]

    for response in tqdm(rejects):
        meta = response["_Meta"]
        print(f"== {meta['HITId']}/{meta['AssignmentId']} ==")
        print("=== Worker Output ===")
        pprint(response)
        print("=== Rejection Email ===")
        print(exp.helper.rejection_email(response, char_limit=9999))
        print()
        confirmation = force_user_input(f"We are about to reject {meta['AssignmentId']}. "
                                        "Please confirm (r)eject, (a)pprove, (s)kip: ",
                                        ["r", "a", "s"])
        if confirmation == "a":
            meta["ShouldApprove"] = True
            # Undo the qualification update in the rejection.
            if meta["AssignmentStatus"] == "Rejected":
                # Make sure we update qualifications
                meta["QualificationUpdated"] = None
            meta["QualityControlDecisions"] = [
                QualityControlDecision(
                    should_approve=True,
                    short_reason="Approved",
                    reason="Approved",
                    qualification_value=71,
                    )
            ]
        elif confirmation == "s":
            meta["ShouldApprove"] = None
        # TODO: support for custom rejection messages.

    # Save the output
    exp.storel("outputs.jsonl", outputs)

    total = sum(len(output) for output in outputs)
    total_accepts = sum(1 for responses in outputs for response in responses
                        if response["_Meta"]["ShouldApprove"] is True)
    total_rejects = sum(1 for responses in outputs for response in responses
                        if response["_Meta"]["ShouldApprove"] is False)
    total_undecided = sum(1 for responses in outputs for response in responses
                          if response["_Meta"]["ShouldApprove"] is None)
    logger.info(f"""Summary:
- Accepts: {total_accepts}
- Rejects: {total_rejects}
- Undecided: {total_undecided}
- Total: {total}""")


def do_pay(args):
    # 0. Find experiment dir.
    exp = Experiment(args.root).find(args.idx)

    def _is_pending(response):
        if "ShouldApprove" not in response["_Meta"]:
            return False
        if response["_Meta"]["AssignmentStatus"] == "Submitted":
            return True
        # Anyone with an approved assignment who is now going to be
        # rejected?
        if response["_Meta"]["AssignmentStatus"] == "Accepted" and not response["_Meta"]["ShouldApprove"]:
            return True
        # Anyone who is still waiting on a bonus
        if response["_Meta"].get("Bonus", 0) > 0 and not response["_Meta"]["BonusPaid"]:
            return True

    # Get summary of task.
    outputs = exp.loadl("outputs.jsonl")
    pending_assns = [response for responses in outputs for response in responses if _is_pending(response)]
    total_accepts = sum(1 for response in pending_assns if response["_Meta"]["ShouldApprove"] and response["_Meta"]["AssignmentStatus"] == "Submitted")
    total_rejects = sum(1 for response in pending_assns if not response["_Meta"]["ShouldApprove"] and response["_Meta"]["AssignmentStatus"] == "Submitted")
    total_bonus = sum(response["_Meta"]["Bonus"] for response in pending_assns)

    # We're going to do something...
    print(f"""Summary:
- Approvals:  {total_accepts} (${total_accepts * get_reward(exp.config):0.2f}),
- Rejections: {total_rejects},
- Bonuses: ${total_bonus}""")

    if True or total_accepts or total_rejects or total_bonus:
        if force_user_input("Are you sure you want to continue? ", ["y", "n"]) == "n":
            sys.exit(1)
        pay_task(exp, use_prod=args.prod)


def do_stop(args):
    # 0. Find experiment dir.
    exp = Experiment(args.root).find(args.idx)
    hits = exp.loadl('hits.jsonl')

    conn = botox.get_client(args.prod)
    for hit in tqdm(hits, desc="Stopping hits"):
        botox.stop_hit(conn, hit)
        exp.storel('hits.jsonl', hits)


def do_clear(args):
    # 0. Find experiment dir.
    exp = Experiment(args.root).find(args.idx)
    hits = exp.loadl('hits.jsonl')

    conn = botox.get_client(args.prod)
    for hit in tqdm(hits, desc="Deleting hits"):
        try:
            if 'DeletedAt' not in hit:
                botox.delete_hit(conn, hit)
            exp.storel('hits.jsonl', hits)
        except Exception as e:
            logger.exception(e)
            continue

    # Move hits.jsonl and outputs.jsonl into an archived directory.
    i = 0
    while exp.exists('.bk-{}'.format(i)):
        i += 1
    logger.info("Backing up data to .bk-%d", i)

    exp.ensure_exists('.bk-{}'.format(i))
    _run(f"mv {exp.path('hits.jsonl')} {exp.path(f'.bk-{i}')}")
    _run(f"mv {exp.path('outputs.jsonl')} {exp.path(f'.bk-{i}')}")


def do_export(args):
    exp = Experiment(args.root).find(args.idx)
    export_task(exp)


def do_metrics(args):
    exp = Experiment(args.root).find(args.idx)
    aggregate_task(exp)
    metrics = compute_metrics(exp)
    # Print metrics.
    print("=== Metrics ===")
    yaml.safe_dump(metrics, sys.stdout)


def do_create_qual(args):
    exp = Experiment(args.root).find(args.idx)

    if exp.config.get('QualificationTypeId'):
        logger.fatal(f"""Already have a qualification type ({exp.config['QualificationTypeId']}) for this
                HIT. If you are sure you want to create a new one,
                delete the existing QualificationTypeId from the
                properties""")
        sys.exit(1)

    conn = botox.get_client(args.prod)
    exp.config['QualificationTypeId'] = botox.create_qualification(
        conn,
        name="Qualification for '{}'".format(exp.config['Title']),
        keywords=exp.config['Keywords'],
        description=exp.config['Description'],
        auto_granted=args.auto_granted,
        auto_granted_value=args.auto_granted_value,
    )
    if 'Qualifications' not in exp.config:
        exp.config['Qualifications'] = []
    exp.config['Qualifications'].append("{} > {}".format(exp.config['QualificationTypeId'], args.cutoff))
    exp.store('task.yaml', exp.config)


# region: manual intervention
def do_set_qual(args):
    exp = Experiment(args.root).find(args.idx)

    conn = botox.get_client(args)
    for worker_id in args.worker_ids:
        current_score = botox.get_qualification(conn, exp.config["QualificationTypeId"], worker_id)
        if args.update or current_score is None:
            botox.set_qualification(conn, exp.config["QualificationTypeId"], worker_id, args.value)
            logger.info(f"Assigned qualification score of {args.value} of {worker_id}")
        else:
            logger.info(f"{worker_id} already has a qualification score of {current_score}")


# def do_query(args):
#     exp = Experiment(args.root).find(args.idx)
#     props = exp.load('hit_properties.json')
#     inputs = exp.loadl('inputs.jsonl')
#     hits = exp.loadl('hits.jsonl')
#     outputs = exp.loadl('outputs.jsonl')
#
#     conn = botox.get_client(args)
#
#     extract_schema = args.extract and jt.parse_schema(args.extract)
#
#     if args.worker_id:
#         print("### Worker {}".format(args.worker_id))
#         print("=== Responses")
#
#         responses = data.get_hits_for_worker(inputs, hits, outputs, args.worker_id)
#         for resp in responses:
#             print("==== HIT: {}".format(resp["HIT"]["HITId"]))
#             if extract_schema:
#                 resp = jt.apply_schema(extract_schema, resp)
#             pprint(resp)
#
#     elif args.hit_id:
#         resp = data.get_hit(inputs, hits, outputs, args.hit_id)
#         print("== Information about HIT {}".format(args.hit_id))
#         if extract_schema:
#             resp = jt.apply_schema(extract_schema, resp)
#         pprint(resp)
#
#     elif args.assn_id:
#         resp = data.get_assignment(inputs, hits, outputs, args.assn_id)
#         print("== Information about assignment {}".format(args.assn_id))
#         if extract_schema:
#             resp = jt.apply_schema(extract_schema, resp)
#         pprint(resp)
#
#     elif args.filter:
#         for resp in data.get_assignments(inputs, hits, outputs, args.filter):
#             print("== Information about assignment {}".format(resp["AssignmentId"]))
#             if extract_schema:
#                 resp = jt.apply_schema(extract_schema, resp)
#             pprint(resp)
#
#
# def do_approve(args):
#     exp = Experiment(args.root).find(args.idx)
#     props = exp.load('hit_properties.json')
#     inputs = exp.loadl('inputs.jsonl')
#     hits = exp.loadl('hits.jsonl')
#     outputs = exp.loadl('outputs.jsonl')
#
#     _, hit, response = data.get_assignment(inputs, hits, outputs, args.assignment_id)
#
#     conn = botox.get_client(args)
#
#     botox.approve_assignment(conn, response, override=True)
#     hit['NumberOfAssignmentsApproved'] += 1
#     botox.set_qualification(conn, props['QualificationTypeId'], response["WorkerId"], 100)
#     exp.storel('hits.jsonl', hits)
#     exp.storel('outputs.jsonl', outputs)
#
#
# def do_message(args):
#     conn = botox.get_client(args)
#     botox.message_worker(conn, args.subject, args.message, args.worker_id)
# endregion


def main():
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-C', '--config', type=str, required=False, default=None, help="Meta configuration file.")
    parser.add_argument('-R', '--root', default='.', help="Root directory where experiments are stored.")
    parser.add_argument('-P', '--prod', action='store_true', help="Use production AMT?")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('init', help=do_init.__doc__)
    command_parser.set_defaults(func=do_init)

    command_parser = subparsers.add_parser('new', help=do_new.__doc__)
    command_parser.add_argument('-u', '--update', action='store_true', default=False,
                                help="If set, we will only run the update hook for experiment "
                                "instead of creating one from scratch.")
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.set_defaults(func=do_new)

    command_parser = subparsers.add_parser('view', help='View an experiment')
    command_parser.add_argument('-p', '--port', default=8080, type=int, help="Which port to serve on")
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.set_defaults(func=do_view)

    command_parser = subparsers.add_parser('launch', help='launch an experiment onto the server and turk')
    command_parser.add_argument('-F', '--force', action='store_true', default=False,
                                help="Force changes in reward and estimated time to match properties")
    command_parser.add_argument('-n', '--max-hits', type=int,
                                help="The number of HITs to publish (defaults to 2 in sandbox)")
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.set_defaults(func=do_launch)

    command_parser = subparsers.add_parser('sync', help='Sync local data with AMT server.')
    command_parser.add_argument('-A', '--auto-qc', action='store_true', default=False,
                                help="Automatically does quality control and approves")
    command_parser.add_argument('-B', '--blocking', action='store_true', default=False, help="Runs sync in a loop")
    command_parser.add_argument('-T', '--wait-time', type=int, default=180, help="Runs sync in a loop")
    command_parser.add_argument('-F', '--force', action='store_true', default=False, help="Force a thorough update")
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.set_defaults(func=do_sync)

    command_parser = subparsers.add_parser('check', help='Do quality control and assign accept statuses to HITs')
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.add_argument('-f', '--fake', action='store_true', default=False,
                                help="Don't actually update any files")
    command_parser.set_defaults(func=do_check)

    command_parser = subparsers.add_parser('pay', help='Do quality control and assign accept statuses to HITs')
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.add_argument('-F', '--force-reversal', action='store_true', help="Force the reversal")
    command_parser.set_defaults(func=do_pay)

    command_parser = subparsers.add_parser('metrics', help='Compute stats on experiment')
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.set_defaults(func=do_metrics)

    command_parser = subparsers.add_parser('export', help='Export data')
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.set_defaults(func=do_export)

    command_parser = subparsers.add_parser('stop', help='Stop on going task.')
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.set_defaults(func=do_stop)

    command_parser = subparsers.add_parser('clear', help='Deletes HITs from AMT')
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.set_defaults(func=do_clear)

    command_parser = subparsers.add_parser('create-qual', help='Handle a qualification')
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.add_argument('-g', '--auto-granted', default=True, type=bool,
                                help="Should auto-grant? (default True)")
    command_parser.add_argument('-v', '--auto-granted-value', type=int, default=100, help="Auto-granted value")
    command_parser.add_argument('-x', '--cutoff', type=int, default=90, help="Cutoff value to place in qualifications")
    command_parser.set_defaults(func=do_create_qual)

    # region: manual commands
    command_parser = subparsers.add_parser('set-qual', help='Handle a qualification')
    command_parser.add_argument('-W', '--worker-ids', nargs="+", type=str, required=True, help="Workers to update quals for")
    command_parser.add_argument('-V', '--value', type=int, required=True, help="The value to set for the worker")
    command_parser.add_argument('-U', '--update', action="store_true", help="If true, update the qual")
    command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    command_parser.set_defaults(func=do_set_qual)

    # command_parser = subparsers.add_parser('query', help='Converts an old directory into a new one.')
    # command_parser.add_argument('-W', '--worker-id', type=str, help="Worker id to query")
    # command_parser.add_argument('-H', '--hit-id', type=str, help="HIT id to query")
    # command_parser.add_argument('-A', '--assn-id', type=str, help="Assignment id to query")
    # command_parser.add_argument('-F', '--filter', type=str, help="Expression to filter on")
    # command_parser.add_argument('-E', '--extract', type=str, help="Expression to extract")
    # command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    # command_parser.set_defaults(func=do_query)

    # command_parser = subparsers.add_parser('approve', help='Quickly approve an assignment')
    # command_parser.add_argument('-A', '--assignment_id', type=str, help="Assignment to approve")
    # command_parser.add_argument('idx', type=int, nargs="?", help="If provided, the batch index to use")
    # command_parser.set_defaults(func=do_approve)

    # command_parser = subparsers.add_parser('message', help='Contact a worker')
    # command_parser.add_argument('-s', '--subject', type=str, help="Set subject")
    # command_parser.add_argument('-m', '--message', type=str, help="Message")
    # command_parser.add_argument('-w', '--worker-id', type=str, help="Worker ID to query")
    # command_parser.set_defaults(func=do_message)
    # endregion

    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        # Load the configuration file.
        # We will try to look for the configuration file in the
        # following locations (in order):
        #   - the path specified by args.config
        #   - ./config.yaml
        #   - ../config.yaml (having a directory of experiments is a
        #   fairly common pattern)
        # If a configuration does not exist in any of these locations,
        # we will copy a template configuration over.
        if args.config is not None:
            assert os.path.exists(args.config), f"The specified meta-configuration at {args.config} does not exist."
        elif not os.path.exists("./config.yaml") and os.path.exists("../config.yaml"):
            args.config = "../config.yaml"
        else:
            args.config = "config.yaml"

        if not os.path.exists(args.config):
            logger.info(f"Copying over a default configuration YAML to {args.config}")
            shutil.copy(os.path.join(TEMPLATE_DIR, 'config.yaml'), args.config)
            assert os.path.exists(args.config)
        with open(args.config) as f:
            args.config = yaml.safe_load(f)

        args.func(args)


if __name__ == "__main__":
    main()
