"""
A experiment viewing server.
"""

import os
import sys
import json
import html
import logging
from collections import Counter
from datetime import datetime
from typing import List, Optional, cast, Union

import bottle
from bottle import Bottle, static_file

from jinja2 import Template
from jinja2.filters import htmlsafe_json_dumps

from .experiment import ExperimentBatch
from .util import sanitize
from .commands import get_reward, get_bonus

logger = logging.getLogger(__name__)

PACKAGE_PATH = os.path.dirname(__file__)
TEMPLATE_PATH = os.path.join(PACKAGE_PATH, "viewer")


class ExperimentViewer:
    """
    This ExperimentViewer provides the following REST endpoints for a
    webserver:

    - /tasks/view/                  : the task viewer interface.
    - /workers/view/                : view responses for a specific
                                      workers.
    - /render/?task=i[&assignment=j]: renders a particular task (i).
    """

    def __init__(self, exp: ExperimentBatch, **variables):
        self.exp: ExperimentBatch = exp
        self.config: dict = exp.config
        self.config["Reward"] = get_reward(self.config)
        self.config["Bonus"] = get_bonus(self.config)
        self.inputs: List[dict] = exp.loadl("inputs.jsonl")

        self.hits: List[Optional[dict]] = exp.loadl("hits.jsonl") if exp.exists("hits.jsonl") else [None for _ in self.inputs]
        self.outputs: List[List[dict]] = cast(List[List[dict]], exp.loadl("outputs.jsonl")) if exp.exists("outputs.jsonl") else [[] for _ in self.inputs]
        self.agg: List[Optional[dict]] = exp.loadl("agg.jsonl") if exp.exists("agg.jsonl") else [None for _ in self.inputs]

        with exp.open("index.html") as f:
            self.template = Template(f.read())

        self.variables = variables

    def info(self, task: Optional[int] = None, assignment: Optional[str] = None) -> dict:
        """
        Gets information about a task.
        :param task:
        :param assignment:
        :return:
        """
        def per_task_info(task_: int):
            return {
                "hasAggregated": self.agg and task_ < len(self.agg) and self.agg[task_],
                "responses": [r["_Meta"]["WorkerId"] for r in self.outputs[task_]] if self.outputs else [],
            }

        if task is None:
            worker_ids = Counter([r["_Meta"]["WorkerId"] for responses in self.outputs for r in responses])
            return {"tasks": [per_task_info(i) for i, _ in enumerate(self.outputs)],
                    "workerIds": worker_ids.most_common()}
        elif assignment is None:
            return per_task_info(task)
        else:
            return self._get_assn_response(task, assignment)

    def update(self, task: int, assignment: Optional[int] = None):
        """
        Called when we POST a result from the task.
        This should save a new entry to outputs.
        """
        obj = {key: value for key, value in bottle.request.forms.items()}
        for key in obj:
            if key.endswith("as_json"):
                obj[key] = json.loads(obj[key])

        if assignment is None:
            # Translate the output into a format understood by experiment's
            # parse_response handler.
            obj = {
                    "_Meta": {
                        "AssignmentId": "texpy-view",
                        "HITId": f"task-{task}",
                        "WorkerId": f"texpy-user",
                        "AssignmentStatus": f"Approved",
                        "EditedManually": True,
                        "SubmitTime": None,
                        "AcceptTime": None,
                        "WorkerTime": None,
                        },
                    "Answer": obj
                    }
            self.outputs[task].append(obj)
        else:
            # Just update this bit.
            self.outputs[task][assignment]["Answer"] = obj
            self.outputs[task][assignment]["_Meta"]["EditedManually"] = True

        self.exp.storel("outputs.jsonl", self.outputs)

        return obj

    def _get_assn_response(self, task: int, assignment: str):
        if assignment.isnumeric():
            assignment = int(assignment)
        if isinstance(assignment, int) and assignment >= len(self.outputs[task]):
            raise ValueError(f"Assignment index beyond range (got: {assignment}, expected < {len(self.outputs[task])})")

        # Try to get the output for the task.
        if assignment is not None and isinstance(assignment, int) and assignment >= 0:
            assert self.outputs is not None
            return self.outputs[task][assignment]["Answer"]
        elif assignment is not None and assignment == "agg" and self.agg is not None:
            return self.agg[task]
        else:
            return None

    def render(self, task: int, assignment: Optional[str] = None):
        """
        Render a specific task
        """
        if task >= len(self.inputs):
            raise ValueError("Task index beyond range")

        input_ = self.inputs[task]
        output = self._get_assn_response(task, assignment) if assignment is not None else None

        return self.template.render({
            'input': sanitize(input_),
            'output': output,
            'config': self.exp.config,
            **self.variables
            })


def serve_viewer(exp: ExperimentBatch, port: int = 8080, **variables):
    # Special casing SERVER_URL because we need it to serve resources
    variables["SERVER_URL"] = f"http://localhost:{port}"

    viewer = ExperimentViewer(exp, **variables)
    logger.info("Serving interface with %d inputs and %d outputs", len(viewer.inputs), len(viewer.outputs) if viewer.outputs else 0)

    # Start server.
    app = Bottle()

    def view():
        """
        Viewer for tasks.
        """
        
        with open(os.path.join(TEMPLATE_PATH, "view.html")) as f:
            template = Template(f.read())
        return template.render({
            'exp': exp,
            'frame_height': exp.config.get("FrameHeight", 9000),
            })

    def get_resource(path):
        return static_file(path, root=exp.path('static/'))

    def get_texpy_resource(path):
        return static_file(path, root=TEMPLATE_PATH)

    def strip_path():
        """Strip trailing '/'s from paths"""
        bottle.request.environ['PATH_INFO'] = bottle.request.environ['PATH_INFO'].rstrip('/')

    app.add_hook('before_request', strip_path)
    app.route('/', 'GET', lambda: bottle.redirect('/view'))
    app.route('/view', 'GET', view)
    app.route('/api/task', 'GET', viewer.info)
    app.route('/api/task/<task:int>', 'GET', viewer.info)
    app.route('/api/task/<task:int>/<assignment>', 'GET', viewer.info)
    app.route('/api/render/<task:int>/<assignment>', 'GET', viewer.render)
    app.route('/api/render/<task:int>', 'GET', viewer.render)
    app.route('/api/render/<task:int>', 'POST', viewer.update)
    app.route('/api/render/<task:int>/<assignment:int>', 'POST', viewer.update)
    app.route('/static/<path:path>', 'GET', get_resource)
    app.route('/_static/<path:path>', 'GET', get_texpy_resource)
    bottle.run(app, reloader=True, port=port)


__all__ = ["serve_viewer"]
