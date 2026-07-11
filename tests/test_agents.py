from lorcy_code.agents.context import ModelSwitchError as ContextError
from lorcy_code.agents.errors import ModelSwitchError
from lorcy_code.agents.middleware import ModelSwitchError as MiddlewareError
from lorcy_code.cli.repl import ChatREPL


def test_model_switch_error_has_one_identity():
    assert ContextError is ModelSwitchError
    assert MiddlewareError is ModelSwitchError


def test_repl_accepts_initial_yolo_mode():
    assert ChatREPL(yolo=True).yolo is True
    assert ChatREPL().yolo is False
