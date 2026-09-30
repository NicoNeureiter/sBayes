"""Generate a commented YAML config template from the pydantic config classes.

The template mirrors the structure of `SBayesConfig`, filling in default values and
marking fields as `<REQUIRED>` or `<OPTIONAL>`. The comments come from the attribute
docstrings in `sbayes/config/config.py`, which are not available at runtime and are
therefore harvested from the source with `ast`.

Usage:
    python -m sbayes.config.generate_template
"""
from __future__ import annotations

import ast
import io
import re
import types
import typing

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from sbayes.config.config import BaseConfig, SBayesConfig

COMMENT_COLUMN = 40
"""Column at which the attribute docstrings are placed."""


def ruamel_yaml_dumps(thing: Any) -> str:
    """Serialise a ruamel object to a YAML string, keeping its comments.

    Uses a `YAML` instance rather than a plain dump, because only the round-trip
    dumper writes the comments attached to a `CommentedMap`.

    Args:
        thing: the object to serialise, usually a `CommentedMap`

    Returns:
        The YAML text.
    """
    yml = YAML()
    yml.indent(mapping=4, sequence=4, offset=4)
    out = io.StringIO()
    yml.dump(thing, out)
    return out.getvalue()



def is_config_class(obj: Any) -> bool:
    """Check whether `obj` is a config class, i.e. a subclass of `BaseConfig`."""
    return isinstance(obj, type) and issubclass(obj, BaseConfig)


def is_docstring(node: ast.stmt) -> bool:
    """Check whether an AST node is a bare string expression, i.e. a docstring."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def analyze_class_docstrings(module_file: str | Path) -> dict[str, dict[str, str]]:
    """Collect the attribute docstrings of every class in a Python module.

    Attribute docstrings are the strings written directly below an annotated
    assignment. Python discards them at runtime, so the source is parsed instead.
    Only classes at module level are inspected.

    Args:
        module_file: path to the Python module to analyse

    Returns:
        The docstrings per class and attribute, `{class_name: {attribute: docstring}}`,
        with whitespace collapsed into single spaces.
    """
    root = ast.parse(Path(module_file).read_text(encoding="utf-8"))

    all_docs: dict[str, dict[str, str]] = {}
    for child in root.body:
        if not isinstance(child, ast.ClassDef):
            continue

        all_docs[child.name] = docs = {}
        last: str | None = None
        for statement in child.body:
            if is_docstring(statement):
                if last:  # the class docstring itself has no preceding attribute
                    docs[last] = re.sub(r"\s+", " ", statement.value.value)
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                last = statement.target.id
            else:
                last = None

    return all_docs


def attach_attribute_docstrings(config_module: types.ModuleType) -> None:
    """Store the attribute docstrings of a config module on its classes.

    Sets `__attrdocs__` on every class of the module, which is what
    `BaseConfig.get_attr_doc` reads.

    Args:
        config_module: the imported module whose source is parsed
    """
    for class_name, docs in analyze_class_docstrings(config_module.__file__).items():
        cls = getattr(config_module, class_name, None)
        if cls is not None:
            cls.__attrdocs__ = docs


# --- Building the template -----------------------------------------------------------

def unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Split an annotation into its type and whether it allows None.

    `Optional[X]` and `X | None` both yield `(X, True)`; anything else yields
    `(annotation, False)`. A union of several types other than None is returned
    unchanged, since it has no single type to expand.

    Args:
        annotation: the resolved annotation of a field

    Returns:
        The type without None, and whether None was allowed.
    """
    if typing.get_origin(annotation) not in (typing.Union, types.UnionType):
        return annotation, False

    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    optional = len(args) < len(typing.get_args(annotation))
    return (args[0] if len(args) == 1 else annotation), optional


def template_literal(field: FieldInfo, optional: bool) -> Any:
    """Determine the value to write for a field in the template.

    Uses an explicit default, or a value built by the field's default factory, and
    otherwise marks the field as `<OPTIONAL>` or `<REQUIRED>`.

    Args:
        field: the pydantic field
        optional: whether the field accepts None

    Returns:
        The default value, a nested template, or the `<REQUIRED>`/`<OPTIONAL>` marker.
    """
    if field.default is not PydanticUndefined and field.default is not None:
        return field.default.value if isinstance(field.default, Enum) else field.default

    if field.default_factory is not None:
        default = field.default_factory()
        if default is not None:
            if isinstance(default, BaseConfig):
                return template(type(default))
            if isinstance(default, (list, dict)):
                return default
            return str(default)

    return "<OPTIONAL>" if optional else "<REQUIRED>"


def template(cfg: type[BaseConfig]) -> CommentedMap:
    """Build the template of one config class, as a commented YAML mapping.

    Nested config classes become nested mappings, including optional ones, so that
    every section of the config appears in the template.

    Args:
        cfg: the config class

    Returns:
        The mapping, with the attribute docstrings attached as comments.
    """
    d = CommentedMap()
    for key, field in cfg.model_fields.items():
        annotation, optional = unwrap_optional(field.annotation)

        nested = is_config_class(annotation)
        d[key] = template(annotation) if nested else template_literal(field, optional)

        # A nested section is introduced by its own class docstring, which ruamel
        # would drop if the key also carried an end-of-line comment.
        docstring = cfg.get_attr_doc(key)
        if docstring and not (nested and annotation.__doc__):
            d.yaml_add_eol_comment(key=key, comment=docstring, column=COMMENT_COLUMN)

    if cfg.__doc__:
        d.yaml_set_start_comment(cfg.__doc__)

    return d


def indent_comments(yaml_str: str) -> str:
    """Indent standalone comments to the level of the section they introduce.

    ruamel writes the comment of a nested mapping at the indentation of its parent;
    this moves it in by one level and adds a blank line before each section, so that
    the template stays readable. The comment of the top-level class stays where it is.

    Args:
        yaml_str: the YAML text as written by ruamel

    Returns:
        The reformatted text.
    """
    lines: list[str] = []
    for line in yaml_str.split("\n"):
        if line.startswith("#") and lines:
            previous = lines[-1]
            indent = len(previous) - len(previous.lstrip())
            line = " " * (4 + indent) + line
        elif line.endswith(":"):
            lines.append("")

        lines.append(line)

    return "\n".join(lines)


def generate_template(config_module: types.ModuleType) -> str:
    """Generate the commented YAML config template of a config module.

    Args:
        config_module: the module defining the config classes, i.e.
            `sbayes.config.config`

    Returns:
        The template as YAML text.
    """
    attach_attribute_docstrings(config_module)
    return indent_comments(ruamel_yaml_dumps(template(SBayesConfig)))


def main(output_path: Path = Path("config_template.yaml")) -> None:
    """Write the config template to a file."""
    from sbayes.config import config

    output_path.write_text(generate_template(config), encoding="utf-8")
    print(f"Wrote the config template to {output_path}")


if __name__ == "__main__":
    main()