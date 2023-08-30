"""
This script generates yaml files with comments from the config classes defined in
sbayes.config.config and attribute docstrings. This requires some code introspection,
which is a bit obscure and at this point not well documented.
"""

import io
from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_core import PydanticUndefined

from ruamel import yaml as yaml

from sbayes.config.config import BaseConfig, SBayesConfig


def ruamel_yaml_dumps(thing):
    y = yaml.YAML()
    y.indent(mapping=4, sequence=4, offset=4)
    out = io.StringIO()
    y.dump(thing, stream=out)
    out.seek(0)
    return out.read()


def generate_template(config_module):
    import ast
    import re

    def is_config_class(obj: Any) -> bool:
        """Check whether the given object is a subtype of BaseConfig"""
        return isinstance(obj, type) and issubclass(obj, BaseConfig)

    def could_be_a_docstring(node) -> bool:
        """Check whether the AST node is a string constant, i.e. could be a doc-string."""
        return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)

    def analyze_class_docstrings(modulefile: str) -> dict:
        """Collect all doc-strings of attributes in each class in a nested dictionary
        of the following structure:
            {class_name: {attribute_name: doc_string}}.

        Args:
            modulefile: the name of the python module to be analysed

        Returns:
            The nested dictionary with attribute doc-strings

        """
        with open(modulefile) as fp:
            root = ast.parse(fp.read())

        alldocs = {}
        for child in root.body:
            if not isinstance(child, ast.ClassDef):
                continue

            alldocs[child.name] = docs = {}
            last = None
            for childchild in child.body:
                if could_be_a_docstring(childchild):
                    if last:  # Skip class doc string
                        s = childchild.value.s
                        # replace multiple spaces and linebreaks by single space.
                        s = re.sub('\\s+', ' ', s)
                        docs[last] = s
                elif isinstance(childchild, ast.AnnAssign):
                    last = childchild.target.id
                else:
                    last = None

        return alldocs

    # Collect all attribute docstring in __attrdocs__ of the corresponding class
    # (used for yaml comments later on)
    for class_name, docs in analyze_class_docstrings(config_module.__file__).items():
        cls: type = vars(config_module)[class_name]
        cls.__attrdocs__ = docs

    def template_literal(field: Field, type_annotation: type):
        """Determine the value to use for `field` in the template. I.e. either use an
        explicitly specified default value, generate a default value from a default
        factory or mark the field as <REQUIRED> or <OPTIONAL>."""

        # If there is a default, use it:
        if field.default not in (None, PydanticUndefined):
            if isinstance(field.default, Enum):
                return field.default.value
            else:
                return field.default

        if field.default_factory and field.default_factory() is not None:
            factory = field.default_factory
            if isinstance(factory, type):
                if issubclass(factory, BaseConfig):
                    # if the default factory is itself a Config, return the template
                    return template(factory)
                if issubclass(factory, list) or issubclass(factory, dict):
                    return factory()
            else:
                return str(factory())

        # Otherwise it may be optional or required:
        if 'NoneType' in str(type_annotation):
            assert not field.required
            return '<OPTIONAL>'
        else:
            return '<REQUIRED>'

    def template(cfg: type(BaseConfig)) -> yaml.CommentedMap:
        """Generate a yaml.CommentedMap from the pydantic config class."""
        d = yaml.CommentedMap()
        for key, field in cfg.model_fields.items():
            if is_config_class(field.annotation):
                d[key] = template(field.annotation)
            else:
                d[key] = template_literal(field, cfg.annotations(key))
                docstring = cfg.get_attr_doc(key)
                if docstring:
                    d.yaml_add_eol_comment(key=key, comment=docstring, column=40)

        if cfg.__doc__:
            d.yaml_set_start_comment(cfg.__doc__)

        return d

    def get_indent(line: str) -> int:
        """Count the number of spaces at the start of `line`."""
        return len(line) - len(str.lstrip(line))

    d = template(SBayesConfig)
    s = ruamel_yaml_dumps(d)
    lines = []
    for line in s.split('\n'):
        if line.startswith('#'):
            indent = ' ' * (4 + get_indent(lines[-1]))
            line = indent + line
        elif line.endswith(':'):
            lines.append('')

        lines.append(line)
        # lines.append('')

    yaml_str = '\n'.join(lines)
    return yaml_str


if __name__ == "__main__":
    from sbayes.config import config
    template_str = generate_template(config)
    with open('config_template.yaml', 'w') as yaml_file:
        yaml_file.write(template_str)
