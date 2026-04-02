################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import os
import pytest
import shutil
import subprocess
import sys

from artifact_helpers import compress_output, extract_artifact
from config_helpers import disable_timer_and_hwmonitor, verify_timer_and_hwmonitor


def _run_tensile(args):
    """Run Tensile in a subprocess for full process-level isolation.

    Streams stdout/stderr to the parent process so pytest captures output
    normally.  Raises CalledProcessError on non-zero exit, which pytest
    reports as a test failure with the full subprocess output.
    """
    subprocess.run(
        [sys.executable, "-c",
         "from Tensile.Tensile import Tensile; import sys; Tensile(sys.argv[1:])",
         *args],
        check=True,
    )


def test_config(tensile_args, config, tmpdir, pytestconfig):
    """Full round-trip test: build -> compress -> wipe -> extract -> run.

    Verifies that the compressed artifact contains everything needed for
    the use-cache phase.  Both the build and run phases are executed in
    separate subprocesses to ensure full process-level isolation — no
    shared module-level state (globalParameters, ISA caches, toolchain
    singletons, etc.) can leak between phases.

    Only runs when neither --build-only nor --use-cache is set (i.e.,
    the default local workflow).
    """
    if pytestconfig.getoption("--build-only") or pytestconfig.getoption("--use-cache"):
        pytest.skip("split mode active — use test_config_build or test_config_run")

    output_dir = os.path.join(tmpdir.strpath, "output")
    is_benchmark_timer_config = "benchmark_timer" in os.path.basename(config)

    # Step 1: Build (in subprocess)
    build_args = [config, output_dir, "--build-only", *tensile_args]
    if not is_benchmark_timer_config:
        disable_timer_and_hwmonitor(build_args)
    _run_tensile(build_args)

    # Step 2: Compress
    artifact_path = compress_output(output_dir)

    # Step 3: Delete everything except the artifact
    shutil.rmtree(output_dir)

    # Step 4: Extract
    extract_artifact(artifact_path, output_dir)
    os.remove(artifact_path)

    # Step 5: Run with cache (in subprocess — clean process, no leftover state)
    run_args = [config, output_dir, "--use-cache", *tensile_args]
    if not is_benchmark_timer_config:
        disable_timer_and_hwmonitor(run_args)
    _run_tensile(run_args)

    # Step 6: Verify
    verify_timer_and_hwmonitor(output_dir,
                                expect_disabled=not is_benchmark_timer_config)
