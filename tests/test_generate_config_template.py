from ruamel.yaml import YAML
from sbayes.config.generate_template import generate_template, unwrap_optional, indent_comments
from typing import Optional


def test_generate_template():
    from sbayes.config import config

    text = generate_template(config)
    loaded = YAML(typ="safe").load(text)

    assert {"data", "model", "mcmc", "results"} <= set(loaded)
    # optional sections are expanded, not collapsed to one marker
    assert isinstance(loaded["data"], dict)
    # required and optional fields are distinguished
    assert "<OPTIONAL>" in text and "<REQUIRED>" in text
    # attribute docstrings end up as comments
    assert "#" in text


def test_unwrap_optional():
    assert unwrap_optional(Optional[int]) == (int, True)
    assert unwrap_optional(int | None) == (int, True)
    assert unwrap_optional(int) == (int, False)


def test_indent_comments_keeps_the_leading_comment():
    assert indent_comments("# top\na: 1\n").startswith("# top")