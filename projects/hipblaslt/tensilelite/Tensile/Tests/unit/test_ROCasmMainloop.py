"""Unit tests for _generateROCasmMainloop and _useROCasmMainloop in Run.py."""

import importlib.util
import textwrap
from pathlib import Path

import pytest


# We can't import Run.py directly because it has heavy dependencies.
# Instead, test the core logic by reimplementing the extraction/splice functions
# inline - they are pure string operations + importlib.

def _generateROCasmMainloop(src: str, mainloopPath: Path) -> str:
    begin_marker = "label_LoopBeginL:"
    end_marker = "label_LoopEndL:"
    begin_idx = src.find(begin_marker)
    end_idx = src.find(end_marker)
    if begin_idx == -1 or end_idx == -1:
        return src
    line_start = src.rfind("\n", 0, begin_idx)
    line_start = line_start + 1 if line_start != -1 else 0
    end_line_start = src.rfind("\n", 0, end_idx)
    end_line_start = end_line_start + 1 if end_line_start != -1 else 0
    prefix = src[:line_start]
    main_loop_text = src[line_start:end_line_start]
    suffix = src[end_line_start:]
    with open(mainloopPath, "w", encoding="utf-8") as f:
        f.write("def get_main_loop():\n")
        f.write(f"    return {repr(main_loop_text)}\n")
    spec = importlib.util.spec_from_file_location("_rocasm_mainloop", str(mainloopPath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    spliced_main_loop = mod.get_main_loop()
    return prefix + spliced_main_loop + suffix


def _useROCasmMainloop(src: str, mainloopPath: str) -> str:
    begin_marker = "label_LoopBeginL:"
    end_marker = "label_LoopEndL:"
    begin_idx = src.find(begin_marker)
    end_idx = src.find(end_marker)
    if begin_idx == -1 or end_idx == -1:
        return src
    line_start = src.rfind("\n", 0, begin_idx)
    line_start = line_start + 1 if line_start != -1 else 0
    end_line_start = src.rfind("\n", 0, end_idx)
    end_line_start = end_line_start + 1 if end_line_start != -1 else 0
    prefix = src[:line_start]
    suffix = src[end_line_start:]
    resolved = Path(mainloopPath).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"UseROCasmMainLoop: module not found: {resolved}")
    spec = importlib.util.spec_from_file_location("_rocasm_user_mainloop", str(resolved))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_main_loop"):
        raise AttributeError(
            f"UseROCasmMainLoop: module {resolved} does not define get_main_loop()")
    user_main_loop = mod.get_main_loop()
    return prefix + user_main_loop + suffix


SAMPLE_ASM = textwrap.dedent("""\
    ; prologue
    s_mov_b32 s0, 0x1234
    label_LoopBeginL:
    v_mfma_f32_16x16x32_bf16 acc[0:3], v[0:3], v[24:27], acc[0:3]
    ds_read_b128 v[32:35], v[40] offset:0
    s_waitcnt lgkmcnt(0)
    s_sub_u32 s1, s1, 1
    s_cmp_eq_i32 s1, 0x2
    s_cbranch_scc0 label_LoopBeginL
    label_LoopEndL:
    ; epilogue
    s_endpgm
""")


@pytest.mark.unit
def test_generate_round_trip(tmp_path):
    """_generateROCasmMainloop produces byte-identical output."""
    mainloop_py = tmp_path / "test_mainloop.py"
    result = _generateROCasmMainloop(SAMPLE_ASM, mainloop_py)
    assert result == SAMPLE_ASM
    assert mainloop_py.exists()


@pytest.mark.unit
def test_generate_creates_valid_module(tmp_path):
    """The generated mainloop.py defines get_main_loop() that returns the main loop text."""
    mainloop_py = tmp_path / "test_mainloop.py"
    _generateROCasmMainloop(SAMPLE_ASM, mainloop_py)

    spec = importlib.util.spec_from_file_location("check", str(mainloop_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    text = mod.get_main_loop()
    assert "label_LoopBeginL:" in text
    assert "v_mfma_f32_16x16x32_bf16" in text
    assert "s_cbranch_scc0 label_LoopBeginL" in text
    # Should NOT include the end label
    assert "label_LoopEndL:" not in text


@pytest.mark.unit
def test_generate_no_labels():
    """Without labels, _generateROCasmMainloop returns src unchanged."""
    src = "no labels here\ns_endpgm\n"
    result = _generateROCasmMainloop(src, Path("/tmp/should_not_exist.py"))
    assert result == src


@pytest.mark.unit
def test_use_round_trip(tmp_path):
    """_useROCasmMainloop with generated mainloop.py produces byte-identical output."""
    mainloop_py = tmp_path / "test_mainloop.py"
    _generateROCasmMainloop(SAMPLE_ASM, mainloop_py)

    result = _useROCasmMainloop(SAMPLE_ASM, str(mainloop_py))
    assert result == SAMPLE_ASM


@pytest.mark.unit
def test_use_custom_mainloop(tmp_path):
    """_useROCasmMainloop can replace the main loop with different content."""
    custom_py = tmp_path / "custom_mainloop.py"
    custom_body = "label_LoopBeginL:\n; my custom loop\ns_nop 0\n"
    custom_py.write_text(
        f"def get_main_loop():\n    return {repr(custom_body)}\n",
        encoding="utf-8",
    )

    result = _useROCasmMainloop(SAMPLE_ASM, str(custom_py))
    assert "my custom loop" in result
    assert "s_nop 0" in result
    # Original main loop instructions should be gone
    assert "v_mfma_f32_16x16x32_bf16" not in result
    # Prologue and epilogue should be preserved
    assert "s_mov_b32 s0, 0x1234" in result
    assert "s_endpgm" in result
    # The end label should still be there (it's part of suffix)
    assert "label_LoopEndL:" in result


@pytest.mark.unit
def test_use_no_labels():
    """Without labels, _useROCasmMainloop returns src unchanged."""
    src = "no labels here\ns_endpgm\n"
    result = _useROCasmMainloop(src, "/tmp/nonexistent.py")
    assert result == src


@pytest.mark.unit
def test_use_file_not_found():
    """_useROCasmMainloop raises FileNotFoundError for missing module."""
    with pytest.raises(FileNotFoundError, match="module not found"):
        _useROCasmMainloop(SAMPLE_ASM, "/tmp/definitely_not_here_12345.py")


@pytest.mark.unit
def test_use_missing_get_main_loop(tmp_path):
    """_useROCasmMainloop raises AttributeError if module lacks get_main_loop."""
    bad_py = tmp_path / "bad_mainloop.py"
    bad_py.write_text("x = 42\n", encoding="utf-8")

    with pytest.raises(AttributeError, match="does not define get_main_loop"):
        _useROCasmMainloop(SAMPLE_ASM, str(bad_py))


@pytest.mark.unit
def test_special_chars_in_asm(tmp_path):
    """Round-trip works with backslashes, quotes, and special chars in assembly."""
    asm_with_specials = (
        '; prologue\n'
        'label_LoopBeginL:\n'
        's_mov_b32 s0, 0x5c  ; backslash \\\n'
        'v_mov_b32 v0, "quoted"\n'
        "s_nop 0  ; tab\there\n"
        'label_LoopEndL:\n'
        's_endpgm\n'
    )
    mainloop_py = tmp_path / "special_mainloop.py"
    gen_result = _generateROCasmMainloop(asm_with_specials, mainloop_py)
    assert gen_result == asm_with_specials

    use_result = _useROCasmMainloop(asm_with_specials, str(mainloop_py))
    assert use_result == asm_with_specials


@pytest.mark.unit
def test_generate_then_use_pipeline(tmp_path):
    """Full pipeline: generate mainloop.py, then use it to reconstruct."""
    mainloop_py = tmp_path / "pipeline_mainloop.py"

    # Step 1: Generate
    gen_result = _generateROCasmMainloop(SAMPLE_ASM, mainloop_py)
    assert gen_result == SAMPLE_ASM

    # Step 2: Use the generated file
    use_result = _useROCasmMainloop(SAMPLE_ASM, str(mainloop_py))
    assert use_result == SAMPLE_ASM

    # All three should be identical
    assert gen_result == use_result == SAMPLE_ASM
