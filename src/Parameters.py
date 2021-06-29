# ************
# File: Parameters.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: class for Venus's parameters.
# ************

import datetime 
from src.utils.SplitStrategy import SplitStrategy
from src.utils.ReluApproximation import ReluApproximation
from src.utils.OSIPMode import OSIPMode

class Logger():
    LOGFILE: str = "venus_log_" + str(datetime.datetime.now()) + ".txt"
    VERBOSITY_LEVEL: int = 0

class Solver():
    # Logging
    logger = Logger()
    # Gurobi time limit per MILP in seconds;
    # Default: -1 (No time limit)
    TIMEOUT: int = -1 
    # Frequency of Gurobi callbacks for ideal cuts
    # Cuts are added every 1 in pow(milp_nodes_solved,IDEAL_FREQ)
    IDEAL_FREQ: float = 1 
    # Frequency of Gurobi callbacks for dependency cuts
    # Cuts are added every 1 in pow(milp_nodes_solved,DEP_FREQ)
    DEP_FREQ: float = 1
    # Whether to use Gurobi's default cuts   
    DEFAULT_CUTS: bool = False
    # Whether to use ideal cuts
    IDEAL_CUTS: bool = True
    # Whether to use inter-depenency cuts
    INTER_DEP_CUTS: bool = True
    # Whether to use intra-depenency cuts
    INTRA_DEP_CUTS: bool = True
    # Whether to use inter-dependency constraints
    INTER_DEP_CONSTRS: bool = True
    # Whether to use intra-dependency constraints
    INTRA_DEP_CONSTRS: bool = True
    # whether to monitor the number of MILP nodes solved and initiate
    # splititng only after BRANCH_THRESHOLD is reached.
    MONITOR_SPLIT: bool = True
    # Number of MILP nodes solved before initiating splitting. Splitting
    # will be initiated only if MONITOR_SPLIT is True.
    BRANCH_THRESHOLD: int = 500
    # Whether to print gurobi output
    PRINT_GUROBI_OUTPUT: bool = False

    def callback_enabled(self):
        """
        Returns 

            True iff the MILP solver is using a callback function.
        """
        if self.IDEAL_CUTS or self.INTER_DEP_CUTS or self.INTRA_DEP_CUTS or self.MONITOR_SPLIT:
            return True
        else:
            return False

    def dep_cuts_enabled(self):
        """
        Returns 

            True iff the MILP solver is using dependency cuts.
        """
        if self.INTER_DEP_CUTS or self.INTRA_DEP_CUTS:
            return True
        else:
            return False

    
class Verifier():
    # Logging
    logger = Logger()
    # number of parallel processes solving subproblems
    VER_PROC_NUM: int = 1

class Splitter():
    # Logging
    logger = Logger()
    # Maximum  depth for node splitting. 
    BRANCHING_DEPTH: int = 7
    # determinines when the splitting process can idle because there are
    # many unprocessed jobs in the jobs queue
    LARGE_N_OF_UNPROCESSED_JOBS: int = 500
    # sleeping interval for when the splitting process idles
    SLEEPING_INTERVAL: int = 3
    # the number of input dimensions still considered to be small
    # so that the best split can be chosen exhaustively
    SMALL_N_INPUT_DIMENSIONS: int = 6
    # splitting strategy
    SPLIT_STRATEGY: SplitStrategy = SplitStrategy.NODE_INPUT
    # the stability ratio weight for computing the difficulty of a problem
    STABILITY_RATIO_WEIGHT: float = 1
    # the parameter of the depth exponent for computing the difficulty of a
    # problem - bigger values encourage splitting
    DEPTH_POWER: float = 1
    # the value of fixed ratio above which the splitting can stop in any
    # case
    STABILITY_RATIO_CUTOFF: float = 0.7
    # the number of parallel splitting processes is 2^d where d is the
    # number of the parameter
    SPLIT_PROC_NUM: int = 0
    # macimum splitting depth
    MAX_SPLIT_DEPTH: int = 1000
    INTER_DEPS = True
    INTRA_DEPS = True

class SIP():
    def __init__(self):
        # relu approximation
        self.RELU_APPROXIMATION = ReluApproximation.MIN_AREA
        # whether to use osip for convolutional layers
        self.OSIP_CONV = OSIPMode.SPLIT
        # number of optimised nodes during osip for convolutional layers
        self.OSIP_CONV_NODES = 200
        # whether to use osip for fully connected layers
        self.OSIP_FC = OSIPMode.SPLIT
        # number of optimised nodes during osip for fully connected
        self.OSIP_FC_NODES = 2
        # osip timelimit in seconds
        self.OSIP_TIMELIMIT = 3

    def is_osip_enabled(self):
        return self.OSIP_CONV == OSIPMode.ON  or self.OSIP_FC == OSIPMode.ON

    def is_split_osip_enabled(self):
        return self.OSIP_CONV == OSIPMode.SPLIT  or self.OSIP_FC == OSIPMode.SPLIT

    def is_osip_conv_enabled(self):
        return self.OSIP_CONV == OSIPMode.ON

    def is_osip_fc_enabled(self, depth=None):
        return self.OSIP_FC == OSIPMode.ON

    def copy(self):
        sip_params = SIP()
        sip_params.RELU_APPROXIMATION = self.RELU_APPROXIMATION
        sip_params.OSIP_CONV = self.OSIP_CONV
        sip_params.OSIP_CONV_NODES = self.OSIP_CONV_NODES
        sip_params.OSIP_FC = self.OSIP_FC
        sip_params.OSIP_FC_NODES = self.OSIP_FC_NODES
        sip_params.OSIP_TIMELIMIT = self.OSIP_TIMELIMIT

        return sip_params


class Params:
    """
    Venus's Parameters
    """

    def __init__(self):
        """
        """
    logger = Logger()
    solver = Solver()
    splitter = Splitter()
    verifier = Verifier()
    sip = SIP()

