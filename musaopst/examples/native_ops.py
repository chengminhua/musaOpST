# SPDX-License-Identifier: MulanPSL-2.0
"""Explicit native protocol descriptor. Arbitrary .mu/.so files are not auto-callable."""
from pathlib import Path


def vector_add(case, inputs, workdir):
    return {"protocol": "vector_add_f32_v1",
            "source": str(Path(__file__).with_name("native_vector_add.mu"))}
