# ************
# File: SplitStrategy.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Enum class for split strategies.
# ************

from enum import Enum

class SplitStrategy(Enum):
    INPUT = "input"
    NODE = "node"
    NODE_INPUT = "node then input"
    INPUT_NODE = "input then node"
    INPUT_NODE_ALT = "alrernate node input"
    NONE = "no splitting"
    class NodeSplitStrategy(Enum):
        ONE_SPLIT = "one split per dependency graph"
        MULTIPLE_SPLITS = "multiple splits per dependency graph"
