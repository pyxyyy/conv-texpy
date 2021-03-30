// The tex.py interface

class Interface {
  constructor() {
    this.tasks = [];
    this.workerIds = [];
    this.taskIdx = 0;

    // Only set when we filter by something.
    this.workerId = null;
    // Which tasks we are going to use (note: this is a pointer to tasks and only contains indices)
    this.selectedTasks = null;
  }

  /**
   * Setups up the interface within the document.
   */
  setUp() {
    // Set up task button handlers.
    $("#task-id").on('change', evt => this.setTask(evt.target.value - 1));
    $("#task-prev").on('click', evt => this.setTask(this.taskIdx - 1));
    $("#task-next").on('click', evt => this.setTask(this.taskIdx + 1));

    // Start with the first task.
    // Get worker IDs and populate list.
    this.refresh();
  }

  /**
   * Updates the list of worker ids in the interface, with links.
   * @param callback
   */
  refresh() {
    const self = this;
    const workerIdList = $("#filter-worker-list");

    function addItem(txt, id, active) {
      const ret = $('<a class="dropdown-item" role="button" href="#" />')
          .text(txt);
      ret
          .on("click", () => {
            workerIdList.find(".dropdown-item").removeClass("active");
            ret.addClass("active");
            self.filterByWorkerId(id);
          });

      if (active) ret.addClass("active");

      return ret;
    }

    $.get("/api/task")
      .done(function (data) {
        // data.values contains a list of all the tasks.
        self.tasks = data.tasks;
        self.workerIds = data.workerIds;

        // Update worker id list.
        workerIdList.empty();
        workerIdList.append(addItem("None", null));
        workerIdList.append($('<div class="dropdown-divider"></div>'));
        self.workerIds.forEach(([id, cnt]) => workerIdList.append(addItem(`${id} (${cnt})`, id)));

        self.filterByWorkerId(null);
      })
      .fail(function(_, status) {
        console.error(status);
      });
  }

  /**
   * Filters the set of tasks (and responses) by worker id.
   * @param workerId
   */
  filterByWorkerId(workerId) {
    console.log(workerId);

    // Set worker id
    this.workerId = workerId;

    if (this.workerId == null) {
      this.selectedTasks = null;
      $("#task-id").attr("max", this.tasks.length);
      $("#task-max-text").text(`of ${this.tasks.length}`);
    } else {
      this.selectedTasks = [];

      for (let i = 0; i < this.tasks.length; i++) {
        if (this.tasks[i].responses.some(w => w === this.workerId)) this.selectedTasks.push(i);
      }
      $("#task-id").attr("max", this.selectedTasks.length);
      $("#task-max-text").text(`of ${this.selectedTasks.length}`);
    }

    $("#task-id").val(0);
    this.setTask(0);
  }

  /**
   * Loads the interface for a task.
   * Setup assignment responses with task ids.
   * @param taskIdx
   */
  setTask(taskIdx) {
    const self = this;

    // Don't do anything to take you out of bounds.
    if (taskIdx < 0 || taskIdx >= this.tasks.length) return;
    if (this.selectedTasks != null && taskIdx >= this.selectedTasks.length) return;

    // Update our task idx
    this.taskIdx = taskIdx;
    $("#task-id").val(this.taskIdx + 1);

    // Translate task ids.
    if (this.selectedTasks != null) {
      taskIdx = this.selectedTasks[taskIdx];
    }

    // Get the task details.
    const assnSelector = $("#assignment-selection");
    const task = self.tasks[taskIdx];

    // task contains
    //    (a) the number of responses
    //    (b) is there an aggregated result or not
    if (!task.hasOwnProperty("hasAggregated") || !task.hasOwnProperty("responses")) {
      console.error("Invalid response: ", task);
      return;
    }

    // Clear out any existing items.
    assnSelector.find("button").remove();
    let defaultResponse = null;
    if (self.workerId != null) {
      defaultResponse = task.responses.findIndex(r => r === self.workerId);
      if (defaultResponse === -1) {
        defaultResponse = 0;
      }
    } else if (task.hasAggregated) {
      defaultResponse = "agg";
    } else if (task.responses.length > 0) {
      defaultResponse = 0;
    }

    // Add new elements for each response.
    function addButton(txt, responseIdx, workerId) {
      const ret = $("<button type='button' class='btn btn-secondary' />");
      ret.text(txt);
      ret.on("click", () => {
        assnSelector.find("button").removeClass("active");
        ret.addClass("active");
        self.setResponse(taskIdx, responseIdx)
      });
      if (responseIdx === defaultResponse) {
        ret.addClass("active");
      }

      if (workerId != null) {
        ret.attr("data-toggle", "tooltip");
        ret.attr("title", workerId);
        ret.tooltip();
        if (self.workerId === workerId) {
          ret.removeClass("btn-secondary");
          ret.addClass("btn-primary");
        }
      }
      return ret;
    }

    if (task.hasAggregated) {
      assnSelector.append(addButton("A", "agg"));
    }

    for (let i = 0; i < task.responses.length; i++) {
      assnSelector.append(addButton(i+1, i, task.responses[i]));
    }

    // TODO not show with worker id
    assnSelector.append(addButton("N", null));

    // Set the default response.
    self.setResponse(taskIdx, defaultResponse);
  }

  /**
   * Sets a particular response in the interface
   * @param taskIdx
   * @param responseIdx
   * TODO: approved / rejected
   */
  setResponse(taskIdx, responseIdx) {
    // Update the address the iframe points to.

    const rawBtn = $("#view-raw-btn");
    const raw = $("#raw-contents");
    const iframe = $("#iframe")[0];

    if (responseIdx == null) {
      iframe.src = `/api/render/${taskIdx}`;

      rawBtn.addClass("disabled");
      raw.text("");
    } else {
      iframe.src = `/api/render/${taskIdx}/${responseIdx}`;

      rawBtn.removeClass("disabled");
      $.get(`/api/task/${taskIdx}/${responseIdx}`)
        .done(function (data) {
          raw.text(JSON.stringify(data, null, 2));
        })
        .fail(function(_, status) {
          console.error(status);
        });
    }
  }
}

