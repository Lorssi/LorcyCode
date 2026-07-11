from lorcy_code.shared.frontmatter import parse_frontmatter
from lorcy_code.shared.text import get_text_content, mask_api_key
from lorcy_code.tools.shell.output import truncate_output
from lorcy_code.tools.shell.semantics import interpret_command_result


def test_frontmatter_and_text_helpers():
    parsed = parse_frontmatter("---\nname: demo\n---\nbody")
    assert parsed is not None
    assert parsed.frontmatter["name"] == "demo"
    assert parsed.body == "body"
    assert get_text_content([{"type": "text", "text": "hello"}, " world"]) == "hello world"
    assert mask_api_key("abcdefghijkl") == "abcdef...ijkl"


def test_shell_output_and_semantics():
    output = truncate_output("ok")
    assert output.content == "ok"
    assert not output.truncated
    assert not interpret_command_result("git status", 0).is_error
