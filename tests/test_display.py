from lorcy_code.cli import display


class FakeTask:
    def __init__(self):
        self.cancelled = False

    def done(self):
        return False

    def cancel(self):
        self.cancelled = True


class FakeLive:
    def __init__(self, events):
        self.events = events

    def update(self, value):
        self.events.append(("update", value))

    def stop(self):
        self.events.append(("stop", None))


def _install_live(monkeypatch, events):
    task = FakeTask()
    live = FakeLive(events)
    monkeypatch.setattr(display, "_progress_task", task)
    monkeypatch.setattr(display, "_progress_live", live)
    monkeypatch.setattr(display, "_subagent_count", 0)
    monkeypatch.setattr(display, "_subagent_parallel", False)
    display._agent_progress["Explore: test"] = {"done": True, "calls": 1}
    return task


def test_begin_model_output_stops_spinner_before_reasoning(monkeypatch):
    events = []
    task = _install_live(monkeypatch, events)

    display.begin_model_output()
    events.append(("reasoning", "analysis"))

    assert task.cancelled is True
    assert display._progress_live is None
    assert display._progress_task is None
    assert display._agent_progress == {}
    assert next(i for i, event in enumerate(events) if event[0] == "stop") < next(
        i for i, event in enumerate(events) if event[0] == "reasoning"
    )


def test_begin_model_output_is_idempotent(monkeypatch):
    events = []
    _install_live(monkeypatch, events)

    display.begin_model_output()
    display.begin_model_output()

    assert [event for event in events if event[0] == "stop"] == [("stop", None)]


def test_finalize_turn_display_cleans_up_empty_or_reasoning_only_turn(monkeypatch):
    events = []
    task = _install_live(monkeypatch, events)

    display.finalize_turn_display()

    assert task.cancelled is True
    assert display._progress_live is None
    assert display._progress_task is None


def test_visible_output_does_not_stop_active_subagent_progress(monkeypatch):
    events = []
    task = _install_live(monkeypatch, events)
    monkeypatch.setattr(display, "_subagent_count", 1)

    display.begin_model_output()
    display.finalize_turn_display()

    assert task.cancelled is False
    assert display._progress_live is not None
