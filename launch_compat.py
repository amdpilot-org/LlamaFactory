#!/usr/bin/env python3
"""Launcher wrapper that applies Python 3.10 compat shim before calling LLaMA-Factory."""

exec(open("/workspace/LlamaFactory/py310_compat.py").read())

from llamafactory.train.tuner import run_exp

run_exp()
