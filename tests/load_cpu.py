"""Compile repaired operator code with serial CPU launch emulation for regression tests.

The shim executes the device algorithms and host checks. It does not validate
CNCC code generation, device synchronization, or MLU performance.
"""

import os
import re
import tempfile
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
BUILD = tempfile.TemporaryDirectory(prefix="bangc32-cpu-")


def serial_launches(source):
    return re.sub(r"<<<[^>]*>>>", "", source)


def load_operators(names, verbose=False):
    build = Path(BUILD.name)
    headers = sorted(ROOT.glob("*.h"))
    for header in headers:
        (build / header.name).write_text(serial_launches(header.read_text()))

    source = ["#include <torch/extension.h>", "#include <vector>", "#include <cmath>"]
    for header in headers:
        source.append(f'#include "{header.name}"')
    bindings = []
    for name in names:
        text = (ROOT / f"{name}.mlu").read_text()
        if re.search(r"PyRun_|PyGIL|ReferencePatch|findSchemaOrThrow", text):
            raise ValueError(f"operator contains reference modification or dispatch bypass: {name}")
        namespace = "operator_" + re.sub(r"\W", "_", name)
        source += [f"namespace {namespace} {{", serial_launches(text), "}"]
        signatures = re.findall(
            r"(torch::Tensor|std::vector<torch::Tensor>)\s+bang_func\s*\((.*?)\)\s*\{", text, re.S
        )
        for return_type, parameters in signatures:
            types = [re.sub(r"\s*\b\w+\s*$", "", parameter.strip())
                     for parameter in parameters.split(",") if parameter.strip()]
            pointer = f"{return_type} (*)({', '.join(types)})"
            bindings.append(f'm.def("{name}", static_cast<{pointer}>(&{namespace}::bang_func));')
    source += ["PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {", *bindings, "}"]
    translation_unit = build / "operators.cpp"
    translation_unit.write_text("\n".join(source))
    os.environ["MAX_JOBS"] = "1"
    return load(
        name="bangc32_cpu_regression",
        sources=[str(translation_unit)],
        extra_include_paths=[str(ROOT / "tests/cpu_shim"), str(build)],
        extra_cflags=["-O0", "-g0", "-DBANGC_CPU_TEST"],
        build_directory=str(build),
        with_cuda=False,
        verbose=verbose,
    )
