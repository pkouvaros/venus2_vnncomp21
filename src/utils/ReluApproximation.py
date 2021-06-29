# ************
# File: ReLUApproximation.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Enum class for the linear approximation of the lower bound of
# the ReLU function.
# ************

from enum import Enum

class ReluApproximation(Enum):
    """
    Types of dependencies.
    """
    ZERO = 0
    IDENTITY = 1
    PARALLEL = 2
    MIN_AREA = 3
    VENUS_HEURISTIC = 4
    OPT_HEURISTIC = 5
