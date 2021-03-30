r"""
tex.py - an experiment framework for Amazon Mechanical Turk.
============================================================

You are running a suite of experiments with AMT, perhaps to collect lots
of data or do quality assessment.

* The AMT web interface is, to put it mildly, hard to use with rich,
  reactive HTML interfaces. You'd like a tool to help you package an
  HTML template and deploy it to AMT.
* When running experiments, you don't just need to get data back from
  AMT, you need to aggregate their data, identify spammy or bad workers,
  etc. You'd like a tool to help you measure the quality of work on a
  batch, identify outliers in terms of time taken and help you manage
  worker qualifications.

Enter ``tex.py``, a simple Python-based tool to help you solve the above
problems.


------------
Installation
------------

Currently, ``tex.py`` can only be installed through a clone of the git
repository. That said, installation is quite simple:

.. code-block:: bash

    # from git repo
    $> git clone https://github.com/arunchaganty/texpy
    $> pip install -e .
    $> tex.py --help

-----
Usage
-----

Here how you can get started with `tex.py` through a few simple steps.

* **init(ialize):** Run ``tex.py init <InterfaceName>`` to initialize a
  new experiment. This will build the latest version of the interface
  and create a **new** directory for it (see configuration details below
  for more on what is built). If you want to just update the code in an
  existing experiment, run ``tex.py init -u``
* **add data:** For each annotation task, write a single _object_ in
  JSON format into an ``inputs.jsonl`` file. This is a minimal example
  of how you'd construct this file in Python::

      import json
      ...
      # objs is a list of objects
      with open('inputs.jsonl', 'w') as f:
        for obj in objs:
          json.dump(obj, f)
          f.write("\n")
    
* **view locally:** Run ``tex.py view <InterfaceName>`` to see how the
  interface looks with your data locally at http://localhost:8080.
  You can run a few tasks and see the corresponding ``outputs.jsonl`` file
  that is produced. Make sure it contains everything you wanted it to!
* **edit task configuration:** The task configuration is in ``task.py``
  (see below for an example) and defines the title, subtitle, estimated
  time, reward and assignments among other things.
  If you've defined a _schema_ for the output data, tex.py can help you
  aggregate data, etc.
* **launch (in sandbox):** run ``tex.py launch <InterfaceName>`` to
  launch the task on the AMT sandbox. Do the sandbox task at
  https://workersandbox.mturk.com.
* **sync, check and pay (sandbox)**: run ``tex.py sync <InterfaceName>``
  to get the worker responses from MTurk. Run ``tex.py check
  <InterfaceName>`` to run automated tests and set the ``ShouldApprove``
  flag on all the assignments. You can and should also manually inspect
  the worker's output using ``tex.py view <InterfaceName>``. Finally,
  run ``tex.py pay <InterfaceName>`` to ensure that the HITs are paid
  for.
* **clear (the sandbox run)**: run ``tex.py clear <InterfaceName>`` to
  delete the tasks AFTER they have been paid. Note, this is a very
  dangerous command to run in production -- once these HITs have been
  cleared, the data can never be recovered again. Instead, run a new
  ``tex.py init`` for each new experiment.
* **launch (for real):** run ``tex.py -P launch <InterfaceName>`` to
  launch the task for real: it will tell you how much it should cost and
  how much money we have. Also try ``tex.py -P sync`` and ``tex.py -P
  check``.

------------------------
Experiment configuration
------------------------

Each task is configured using a `task.py` template.
"""
