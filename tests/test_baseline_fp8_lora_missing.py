"""Stage0 baseline guard: assert the FP8 LoRA MI300X feature is absent.

This test codifies the Stage0 missing-feature baseline for amdpilot-org/LlamaFactory#1.
It verifies that:
  * ``llamafactory.extras.fp8_lora_mi300x`` does not exist (no entry point).
  * ``quantization_bit`` only accepts ``int`` (4|8) or ``None`` -- ``fp8`` is rejected.
  * The FP8 LoRA example yaml and the backward smoke test are not present.

Stage0 must freeze the baseline at metric ``0.0``. Any of these checks turning
True means the feature leaked in and the baseline is no longer the clean starting point.
The downstream executor will add the FP8 LoRA wiring (entry point, fp8 hparam,
train-smoke config) and flip ``current_has_requested_feature`` to True.
"""

import importlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FP8_MODULE_FILE = REPO / "src" / "llamafactory" / "extras" / "fp8_lora_mi300x.py"
BACKWARD_TEST = REPO / "tests" / "test_fp8_lora_backward_mi300x.py"
EXAMPLE_YAML = REPO / "examples" / "train_lora_fp8_llama33_70b_mi300x.yaml"


def test_fp8_lora_entry_point_absent():
    """The FP8 LoRA entry point module file must not exist at Stage0."""
    assert not FP8_MODULE_FILE.exists(), (
        "Stage0 baseline broken: fp8_lora_mi300x.py entry point already present."
    )


def test_fp8_lora_module_not_importable():
    """Importing the FP8 LoRA entry point must fail at Stage0."""
    try:
        importlib.import_module("llamafactory.extras.fp8_lora_mi300x")
        raise AssertionError("fp8_lora_mi300x unexpectedly imported -- feature present.")
    except ModuleNotFoundError:
        return


def test_quantization_bit_rejects_fp8():
    """``quantization_bit`` is ``int | None`` and rejects the string ``fp8``.

    The field reads ``quantization_bit: int | None = field(...)``. Because the
    annotation is ``int | None``, the string ``fp8`` is rejected (it is not an
    integer), and the only valid bits are 4 / 8 / None.
    """
    src = (REPO / "src" / "llamafactory" / "hparams" / "model_args.py").read_text()
    assert (
        "quantization_bit: int | None = field(" in src
    ), "quantization_bit annotation is no longer `int | None` (fp8 may have leaked in)"
    # The FinetuningArguments assert also pins to 4/8 only.
    fa = (REPO / "src" / "llamafactory" / "hparams" / "finetuning_args.py").read_text()
    assert "[None, 8, 4]" in fa, "ref_model_quantization_bit assert changed"


def test_fp8_lora_backward_smoke_absent():
    """The FP8 LoRA backward smoke test must not exist at Stage0."""
    assert not BACKWARD_TEST.exists(), (
        "Stage0 baseline broken: backward smoke test already present."
    )


def test_fp8_lora_example_yaml_absent():
    """The FP8 LoRA training example yaml must not exist at Stage0."""
    assert not EXAMPLE_YAML.exists(), (
        "Stage0 baseline broken: example yaml already present."
    )


if __name__ == "__main__":
    tests = [
        ("test_fp8_lora_entry_point_absent", test_fp8_lora_entry_point_absent),
        ("test_fp8_lora_module_not_importable", test_fp8_lora_module_not_importable),
        ("test_quantization_bit_rejects_fp8", test_quantization_bit_rejects_fp8),
        ("test_fp8_lora_backward_smoke_absent", test_fp8_lora_backward_smoke_absent),
        ("test_fp8_lora_example_yaml_absent", test_fp8_lora_example_yaml_absent),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS " + name)
        except AssertionError as exc:
            print("FAIL " + name + ": " + str(exc))
            failures += 1
    if failures:
        print(str(failures) + " test(s) FAILED -- Stage0 baseline broken")
        raise SystemExit(1)
    print("ALL TESTS PASSED -- Stage0 missing-feature baseline confirmed")
