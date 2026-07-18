from __future__ import annotations

import ast
import re
from pathlib import Path


TAG = re.compile(
    r"^# Feature: binary-llm-conversion-framework, Property (\d+): (.+)$"
)

EXPECTED_TITLES = (
    "Claim traceability and scale-qualified evidence",
    "Experiment-family isolation",
    "Seal mismatch detection and immutable lineage",
    "Monotonic scale promotion",
    "Exact role-based binary scope",
    "Reproduction classification follows source differences",
    "Stage 1 trainability and scale shape",
    "Non-finite state fails closed",
    "Ambiguity resolution is explicit and scoped",
    "Matched comparisons change only declared factors",
    "Progressive function and derivative match the reference mathematics",
    "Progressive partition coverage and schedule determinism",
    "Dual-scale algebra and update semantics",
    "Representation identity captures inference exceptions",
    "Sign substitution cannot bypass gates",
    "Recovery keeps the target operator and routes teachers correctly",
    "Recovery allocation and progress policy",
    "Provenance-complete, leakage-free corpora",
    "Release slices retain trustworthy support",
    "Run evidence is uniquely attributable and reproducible",
    "Paired statistics preserve semantic independence",
    "Evidence and blind-test status are temporal invariants",
    "Reports cannot hide failed slices",
    "Iris metric aggregation is complete and bounded",
    "Iris release and continuation thresholds are exact",
    "Budget exhaustion stops unsupported methods",
    "Exact artifact accounting reconciles to disk",
    "Binary packing is an exact round trip",
    "Export provenance and tolerance preregistration",
    "Device gates are deterministic postprocessing",
    "Device evidence is complete",
    "Release selection is fail-closed and size-optimal",
    "Promotion reports are complete and retain oracle identity",
)


def test_property_inventory_has_exact_tags_and_hypothesis_settings() -> None:
    root = Path(__file__).parent
    files = tuple(sorted(root.glob("test_binary_llm_properties_*.py")))
    records: dict[int, tuple[str, str]] = {}
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        tree = ast.parse("\n".join(lines), filename=str(path))
        functions = {
            node.lineno: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for line_number, line in enumerate(lines, start=1):
            match = TAG.fullmatch(line)
            if match is None:
                continue
            number = int(match.group(1))
            assert number not in records, f"duplicate Property {number}"
            following = [
                (function_line, node)
                for function_line, node in functions.items()
                if function_line > line_number
            ]
            assert following, f"Property {number} has no following test"
            _, function = min(following, key=lambda item: item[0])
            assert function.name.startswith("test_"), f"Property {number} tag is not near a test"
            decorators = function.decorator_list
            assert any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "given"
                for decorator in decorators
            ), f"Property {number} lacks @given"
            settings_calls = [
                decorator
                for decorator in decorators
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "settings"
            ]
            assert len(settings_calls) == 1, f"Property {number} requires one @settings"
            max_examples = next(
                (
                    keyword.value.value
                    for keyword in settings_calls[0].keywords
                    if keyword.arg == "max_examples"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, int)
                ),
                None,
            )
            assert max_examples is not None and max_examples >= 100, (
                f"Property {number} max_examples must be >= 100"
            )
            records[number] = (match.group(2), path.name)
    assert set(records) == set(range(1, 34))
    assert len(records) == 33
    assert tuple(records[number][0] for number in range(1, 34)) == EXPECTED_TITLES
