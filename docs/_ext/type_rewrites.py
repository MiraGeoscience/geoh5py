# docs/_ext/type_rewrites.py
import re

from sphinx_autodoc_typehints import process_signature

# compile once
_DEF_PATTERNS = [
    (re.compile(r'\bnp\.ndarray\b'), 'numpy.ndarray'),
    (re.compile(r'[^.]\bUUID\b'),        'uuid.UUID'),
    # add more: (re.compile(r'\bPD\.DataFrame\b'), 'pandas.DataFrame'), ...
]

def _rewrite_types(lines: list[str]) -> None:
    """In-place rewrite of type names inside docstrings."""
    text = "\n".join(lines)
    print(f"HACK IN: {text}")
    for pat, repl in _DEF_PATTERNS:
        text = pat.sub(repl, text)
        print(f"HACK MID: {text}")
    # write back
    lines[:] = text.splitlines()

def _process_signature(
    app,
    what: str,
    name: str,
    obj,
    signature: str,
    options,
    return_annotation: str,
):
    r = process_signature(app, what, name, obj, signature, options, return_annotation)
    if r is not None:
        print(f"HACK PRE: {r[0]}")
        sig_lines = r[0].splitlines()
        _rewrite_types(sig_lines)
        replaced_text = "\n".join(sig_lines)
        if replaced_text != r[0]:
            print(f"HACK OUT: {replaced_text}")
            modified = list(r)
            modified[0] = replaced_text
            r = tuple(modified)
    return r


def _process_docstring(app, what, name, obj, options, lines):
    #if not name.startswith("geoh5py."):
    #    return

    # Only touch things that have parameter/return sections etc.
    # (classes, functions, methods, attributes)
    if what in {"class", "function", "method", "attribute"}:
        _rewrite_types(lines)


def setup(app):
    app.add_config_value("always_document_param_types", False, "html")
    app.add_config_value("typehints_fully_qualified", False, "env")
    app.add_config_value("typehints_document_rtype", True, "env")
    app.add_config_value("typehints_document_rtype_none", True, "env")
    app.add_config_value("typehints_use_rtype", True, "env")
    app.add_config_value("typehints_defaults", None, "env")
    app.add_config_value("simplify_optional_unions", True, "env")
    app.add_config_value("always_use_bars_union", False, "env")
    app.add_config_value("typehints_formatter", None, "env")
    app.add_config_value("typehints_use_signature", True, "env")  # False ?
    app.add_config_value("typehints_use_signature_return", True, "env")  # False ?
    app.add_config_value("typehints_fixup_module_name", None, "env")
    #app.add_role("sphinx_autodoc_typehints_type", sphinx_autodoc_typehints_type_role)

    # Run early so rewrites happen *before* napoleon/numpydoc parse the text
    app.connect("autodoc-process-signature", _process_signature, priority=100)
    #app.connect("autodoc-process-docstring", _process_docstring, priority=100)
    return {"version": "0.1", "parallel_read_safe": True}
