# ************
# File: Verifier.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Main verification process.
# ************

import multiprocessing as mp
import queue
from timeit import default_timer as timer
from src.NeuralNetwork import NeuralNetwork
from src.VerificationProblem import VerificationProblem
from src.VerificationProcess import VerificationProcess
from src.SplitProcess import SplitProcess
from src.InputSplitter import InputSplitter
# from src.utils.Logger import get_logger
from src.utils.SplitStrategy import SplitStrategy
from src.utils.SolveResult import SolveResult
from src.utils.SolveReport import SolveReport
from src.utils.SplitReport import SplitReport
from src.utils.VerificationReport import VerificationReport
from src.MILPSolver import MILPSolver

class Verifier:
    
    # logger = None

    def __init__(self, nn, spec, params):
        """
        Arguments:

            nn: NeuralNetwork

            spec: Specification

            params: Parameters
        """
        self.prob = VerificationProblem(nn, spec, 0, params.logger.LOGFILE)
        self.prob.bound_analysis(params.sip)
        self.params = params
        self.reporting_queue = mp.Queue()
        self.jobs_queue = mp.Queue()
        # if Verifier.logger is None:
            # Verifier.logger = get_logger(__name__, params.logger.LOGFILE)
        self.split_procs = []
        self.ver_procs = []


    def verify(self, complete=True):
        """
        Verifies a neural network against a specification.

        Arguments:

            complete: bool expressing whether to do complete or incomplete
            verification. In complete verification the verification problem is
            translated into MILP. In incomplete verification the verification
            problem is attempted to be solved using symbolic interval
            propagation.

        Returns:

            VerificationReport
        """
        # Verifier.logger.info(f'Verifying {self.prob.to_string()}')
        # if complete == True:
            # ver_report = self.verify_complete()
        # else:
            # ver_report = self.verify_incomplete()
        ver_report = self.verify_complete()
        # Verifier.logger.info('Verification completed')
        # Verifier.logger.info(ver_report.to_string())

        return ver_report



    # def verify_incomplete(self, prob):
        # """
        # Attempts to solve a verification problem using symbolic interval propagation.

        # Returns:

            # VerificationReport
        # """
        # start = timer()
        # ver_report = VerificationReport(self.params.logger.LOGFILE)
        # prob.bound_analysis(self.params.sip)
        # # SIP analysis
        # if prob.satisfies_spec(self.params.sip):
            # ver_report.result = SolveResult.SATISFIED 
            # Verifier.logger.info('Verification problem is satisfied from bound analysis.')
        # else:
            # # LP analysis
            # slv = MILPSolver(prob, self.params.solver, self.params.sip, lp=True) 
            # slv_report =  slv.solve()
            # if slv_report.result == SolveResult.SATISFIED:
                # ver_report.result = SolveResult.SATISFIED 
                # Verifier.logger.info('Verification problem is satisfied from LP analysis.')
            # elif slv_report.result == SolveResult.UNSATISFIED:
                # nn_out = prob.nn.predict(slv_report.cex)
                # if not prob.spec.is_satisfied(nn_out, nn_out):
                    # ver_report.result = SolveResult.UNSATISFIED
                    # ver_report.cex = slv_report.cex
                    # Verifier.logger.info('Verification problem is not satisfied from LP analysis.')
                # else: 
                    # ver_repor.result = SolveResult.UNKNOWN
                    # Verifier.logger.info('Verification problem could not be solved via incomplete verification.')

        # ver_report.runtime = timer() - start
        
        # return ver_report


    def verify_incomplete(self):
        """
        Attempts to solve a verification problem using symbolic interval propagation.

        Returns:

            VerificationReport
        """
        ver_report = VerificationReport(self.params.logger.LOGFILE)
        start = timer()
        self.prob.bound_analysis(self.params.sip)
        if self.prob.satisfies_spec():
            ver_report.result = SolveResult.SATISFIED 
            # print('Verification problem was solved via bound analysis')
            # Verifier.logger.info('Verification problem was solved via bound analysis')
        else:
            ver_report.result = SolveResult.UNKNOWN
            # Verifier.logger.info('Verification problem could not be solved via bound analysis.')

        ver_report.runtime = timer() - start

        return ver_report
    
       
    def verify_complete(self):
        """
        Solves a verification problem by solving its MILP representation.

        Arguments:
            
            prob: VerificationProblem

        Returns:

            VerificationReport
        """
        start = timer()
        # try to solve the problem using the bounds and lp
        report = self.verify_incomplete()
        if report.result == SolveResult.SATISFIED:
            self.terminate_procs()
            return report
    
        # start the splitting and worker processes
        self.generate_procs()
        # read results
        ver_report = self.process_report_queue()
        self.terminate_procs()

        ver_report.runtime = timer() - start



        return ver_report

    def process_report_queue(self):
        """ 
        Reads results from the reporting queue until encountered an UNSATISFIED
        result, or until all the splits have completed

        Returns

            VerificationReport
        """
        start = timer()
        ver_report = VerificationReport(self.params.logger.LOGFILE)
        while True:
            try:
                time_elapsed = timer() - start
                tmo = self.params.solver.TIME_LIMIT - time_elapsed
                report = self.reporting_queue.get(timeout=tmo)
                if isinstance(report, SplitReport):
                    ver_report.process_split_report(report)
                elif isinstance(report, SolveReport):
                    ver_report.process_solve_report(report)
                    if report.result == SolveResult.BRANCH_THRESHOLD:
                        print('Threshold of MIP nodes reached. Turned off monitor split.')
                        # Verifier.logger.info('Threshold of MIP nodes reached. Turned off monitor split.')
                        self.params.solver.MONITOR_SPLIT = False
                        self.generate_procs()
                    elif report.result == SolveResult.UNSATISFIED:
                        # Verifier.logger.info('Read UNSATisfied result. Terminating ...')
                        break
                else:
                        raise Exception(f'Unexpected report read from reporting queue {type(report)}')

                # termination conditions
                if ver_report.finished_split_procs_count == len(self.split_procs) \
                and ver_report.finished_jobs_count >= ver_report.jobs_count:
                    # Verifier.logger.info("All subproblems have finished. Terminating...")
                    if ver_report.timedout_jobs_count == 0:
                        ver_report.result = SolveResult.SATISFIED
                    else:
                        ver_report.result = SolveResult.TIMEOUT
                    break
            except queue.Empty:
                # Timeout occured
                ver_report.result = SolveResult.TIMEOUT
                break
            except KeyboardInterrupt:
                # Received terminating signal
                ver_report.result = SolveResult.INTERRUPTED
                break
                    
        return ver_report


    def generate_procs(self):
        """
        Creates splitting and verification processes.

        Returns:
            
            None
        """
        self.generate_split_procs()
        self.generate_ver_procs()

    def generate_split_procs(self):
        """
        Creates splitting  processes.

        Returns:
            
            None
        """
        if self.params.splitter.SPLIT_STRATEGY != SplitStrategy.NONE \
        and self.params.solver.MONITOR_SPLIT == False:
            if self.params.splitter.SPLIT_PROC_NUM > 0:
                isplitter = InputSplitter(self.prob,
                                          self.prob.stability_ratio,
                                          self.params.splitter.SMALL_N_INPUT_DIMENSIONS,
                                          self.params.splitter.STABILITY_RATIO_CUTOFF,
                                          self.params.splitter.DEPTH_POWER,
                                          self.params.sip,
                                          self.params.logger.LOGFILE)
                splits = isplitter.split_up_to_depth(self.params.splitter.SPLIT_PROC_NUM)
            else:
                splits = [self.prob]
            self.split_procs = [SplitProcess(i+1,
                                             splits[i],
                                             self.params.splitter,
                                             self.params.sip,
                                             self.jobs_queue,
                                             self.reporting_queue)
                                for i in range(len(splits))] 
            for proc in self.split_procs:
                proc.start()
                # Verifier.logger.info(f'Generated {len(self.split_procs)} split processes')
        else:
            self.jobs_queue.put(self.prob)
            # Verifier.logger.info('Added original verification problem to job queue.')

    def generate_ver_procs(self):
        """
        Creates verification  processes.

        Returns:
            
        None
        """
        if self.params.solver.MONITOR_SPLIT == True \
        or self.params.splitter.SPLIT_STRATEGY == SplitStrategy.NONE:
            procs_to_gen = range(1)
        else:
            procs_to_gen = range(len(self.ver_procs), self.params.verifier.VER_PROC_NUM)
           
        ver_procs = [VerificationProcess(i+1,
                                         self.jobs_queue,
                                         self.reporting_queue,
                                         self.params.solver,
                                         self.params.sip)
                     for i in procs_to_gen]
        for proc in ver_procs:
            proc.start()
        self.ver_procs = self.ver_procs + ver_procs    
        # Verifier.logger.info(f'Generated {len(procs_to_gen)} verification processes.')

    def terminate_procs(self):
        """
        Terminates all splitting and verification processes.

        Returns:

            None
        """
        self.terminate_split_procs()
        self.terminate_ver_procs()

    def terminate_split_procs(self):
        """
        Terminates all splitting processes.

        Returns:

            None
        """
        try:
            for proc in self.split_procs:
                proc.terminate()
                proc.join()
                proc.close()
        except:
            raise Exception("Could not terminate splitting processes.")

    def terminate_ver_procs(self):
        """
        Terminates all verification processes.

        Returns:

            None
        """
        try:
            for proc in self.ver_procs:
                proc.terminate()
                proc.join()
                proc.close()
        except:
            raise Exception("Could not terminate verification processes.")
