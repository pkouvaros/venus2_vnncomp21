# ************
# File: SplitProcess.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Split process managing input and node splits.
# ************

from multiprocessing import Process
from src.utils.SplitReport import SplitReport
from src.NodeSplitter import NodeSplitter
from src.InputSplitter import InputSplitter
from src.VerificationProblem import VerificationProblem
# from src.utils.Logger import get_logger
from src.utils.SplitStrategy import SplitStrategy
from src.utils.OSIPMode import OSIPMode
from src.utils.ReluApproximation import ReluApproximation
import time

class SplitProcess(Process):
    
    process_count = 0
    # logger = None

    def __init__(self, id, prob, params, sip_params, jobs_queue, reporting_queue):
        """
        Arguments:
            
            id: int of the identity of the process.

            prob: VerificationProblem.

            params: Parameters.splitter.

            jobs_queue: queue to enqueue the resulting subproblems.

            reporting_queue: queue not enqueue the splitting report.

            logfile: str of path to logfile.
        """
        super(SplitProcess, self).__init__()
        self.id = id
        self.initial_prob = prob
        self.params = params
        self.sip_params = sip_params
        self.jobs_queue = jobs_queue
        self.reporting_queue = reporting_queue
        self.jobs_count = 0
        self.node_split_count = 0
        self.input_split_count = 0
        self.depth=0
        self.split_queue = [self.initial_prob]
        self.process_count += 1
        # if SplitProcess.logger is None:
            # SplitProcess.logger = get_logger(__name__, self.params.logger.LOGFILE)


    def run(self):        
        # SplitProcess.logger.info(f'Running split process {self.id}')
        self.split()
        # SplitProcess.logger.info(f'Split process {self.id} finished')
        self.process_count -= 1
        self.reporting_queue.put(SplitReport(self.id,
                                             self.jobs_count, 
                                             self.node_split_count, 
                                             self.input_split_count))
 

    def split(self):
        """
        Splits the verification problem from the top of the split_queue into a
        set of subproblems.
        """

        while len(self.split_queue) > 0:
            if self.jobs_queue.qsize() >= self.params.LARGE_N_OF_UNPROCESSED_JOBS:
                time.sleep(self.params.SLEEPING_INTERVAL)
            else:
                prob  = self.split_queue.pop()
                split_method = self._select_split_method(prob)
                subprobs = split_method(prob)
                if len(subprobs) > 0:
                    self.process_subprobs(subprobs)
                else:
                    self.add_to_job_queue(prob)

    def _select_split_method(self, prob=None):
        """
        Selects the split methods as per the split parameters.

        Argumens:

            prob: VerificationProblem for which splitting will be performed.

        Returns:

            split method.
        """
        if self.params.SPLIT_STRATEGY == SplitStrategy.INPUT: 
            return self.input_split
        elif self.params.SPLIT_STRATEGY == SplitStrategy.NODE:
            return self.node_split
        elif self.params.SPLIT_STRATEGY == SplitStrategy.NODE_INPUT:
            return self.node_input_split
        elif self.params.SPLIT_STRATEGY == SplitStrategy.INPUT_NODE:
            return self.input_node_split
        elif self.params.SPLIT_STRATEGY == SplitStrategy.INPUT_NODE_ALT:
            if prob.split_strategy == SplitStrategy.INPUT:
                return self.node_input_split
            else:
                return self.input_node_split
        elif self.params.STRATEGY == SplitStrategy.NONE:
            return self.no_split


    def no_split(self, prob):
        """
        Does not carry out any splitting.

        Arguments:
            
            prob: VerificationProblem

        Returns

            empty list
        """
        return []

    def node_input_split(self, prob):
        """
        Does node split. If not successfull it does input split.

        Arguments:
            
            prob: VerificationProblem

        Returns

            list of verification subproblems
        """
        subprobs = self.node_split(prob)
        if len(subprobs) == 0:
            subprobs = self.input_split(prob)

        return subprobs

    def input_node_split(self, prob):
        """
        Does input split. If not successfull it does node split.

        Arguments:
            
            prob: VerificationProblem

        Returns

            list of verification subproblems
        """
        subprobs = self.input_split(prob)
        if len(subprobs) == 0:
            subprobs = self.node_split(prob)

        return subprobs

    def node_split(self, prob):
        """
        Does node split. 

        Arguments:
            
            prob: VerificationProblem

        Returns

            list of verification subproblems
        """
        nsplitter = NodeSplitter(prob,
                                 self.params.BRANCHING_DEPTH,
                                 self.sip_params,
                                 self.params.logger.LOGFILE,
                                 self.params.INTRA_DEPS,
                                 self.params.INTER_DEPS)
        subprobs = nsplitter.split()
        # print(f"Finished node splitting - {len(subprobs)} subproblems")
        # SplitProcess.logger.info(f"Finished node splitting - {len(subprobs)} subproblems")
        if len(subprobs) > 0: self.node_split_count += 1

        return subprobs 
   
    def input_split(self, prob):
        """
        Does input split. 

        Arguments:
            
            prob: VerificationProblem

        Returns

            list of verification subproblems
        """
        isplitter = InputSplitter(prob,
                                  self.initial_prob.stability_ratio,
                                  self.params.SMALL_N_INPUT_DIMENSIONS,
                                  self.params.STABILITY_RATIO_CUTOFF,
                                  self.params.DEPTH_POWER,
                                  self.sip_params,
                                  self.params.logger.LOGFILE)
        subprobs = isplitter.split()
        # SplitProcess.logger.info(f"Finished input splitting - {len(subprobs)} subproblems")
        if len(subprobs) > 0: self.input_split_count += 1
        
        return subprobs


    def add_to_job_queue(self, prob):
        """
        Adds a verification subproblem to job queue.

        Arguments:
            
            prob: VerificationProblem

        Returns:
            
            None
        """
        opt_prob = self.osip_optimise(prob)
        opt_prob = prob
        self.jobs_queue.put(opt_prob)
        self.jobs_count += 1
        # SplitProcess.logger.info(f"Added verification subproblem {prob.id} to job queue")

    def osip_optimise(self, prob):
        if self.sip_params.is_split_osip_enabled():
            sp = self.sip_params.copy()
            if sp.RELU_APPROXIMATION == ReluApproximation.MIN_AREA:
                sp.RELU_APPROXIMATION = ReluApproximation.VENUS_HEURISTIC
            else:
                sp.RELU_APPROXIMATION = ReluApproximation.MIN_AREA
            osip_prob = VerificationProblem(prob.nn.copy(),
                                            prob.spec.copy(),
                                            prob.depth,
                                            self.params.logger.LOGFILE)
            osip_prob.bound_analysis(sp)
            # SplitProcess.logger.info('SIP bounds {:.4f} - OSIP bounds {:.4f}'
                                     # .format(prob.output_range, osip_prob.output_range))
            if osip_prob.output_range < prob.output_range:
                return osip_prob
            else:
                return prob
        else:
            return prob

    def add_to_split_queue(self, prob):
        """
        Adds a verification subproblem to split queue.

        Arguments:
            
            prob: VerificationProblem

        Returns:
            
            None
        """
        self.split_queue = [prob] + self.split_queue
        # SplitProcess.logger.info(f"Added verification subproblem {prob.id} to split queue")

    def process_subprobs(self, probs):
        """
        Enqueues  a list of verification subproblems to the appropriate queue:
        no queue if a subproblem already satisfies the specification;
        jobs_queue if the stability ratio of the subproblem is above the
        threshold; split queue otherwise.

        Arguments:

            probs: list of VerificationProblem

        Returns:

            None
        """
        for prob in probs:
            if prob.satisfies_spec(): 
                pass
                # SplitProcess.logger.info(f'Verfication subproblem {prob.id} discarded as it already satisfies the specification')
            elif prob.stability_ratio >= self.params.STABILITY_RATIO_CUTOFF: 
                self.add_to_job_queue(prob)
            else:
                self.add_to_split_queue(prob)

    @staticmethod
    def has_active_processes():
        """
        Returns:

            bool expresssing whether or not there are currently running
            splitting processs
        """
        return SplitProcess.process_count > 0 
        

