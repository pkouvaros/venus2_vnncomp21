# ************
# File: VerificationProcess.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Process for solving a verification problem.
# ************

from multiprocessing import Process
# from src.utils.Logger import get_logger
from src.MILPSolver import MILPSolver
from src.utils.SolveReport import SolveReport
from src.utils.SolveResult import SolveResult
from timeit import default_timer as timer
import queue

class VerificationProcess(Process):

    # logger = None
    TIMEOUT = 3600

    def __init__(self, id, jobs_queue, reporting_queue, solver_params, sip_params):
        """
        Arguments:

            jobs_queue: queue of verification problems to read from.

            reporting_queue: queue of solve reports to enqueue the verification
            results.

            params: Params.solver

            logfile: str of path to logfile.
        """
        super(VerificationProcess, self).__init__()
        self.id = id
        self.jobs_queue = jobs_queue
        self.reporting_queue = reporting_queue
        self.solver_params = solver_params
        self.sip_params = sip_params
        # if VerificationProcess.logger is None:
            # VerificationProcess.logger = get_logger(__name__ + str(self.id), solver_params.logger.LOGFILE)

    def run(self):
        while True:
            try:
                prob = self.jobs_queue.get(timeout=self.TIMEOUT)
                # VerificationProcess.logger.info('Subprocess {} '
                                                # 'started job {}, '.format(self.id, 
                                                                          # prob.id))
                prob.bound_analysis(self.sip_params)
                # slv_report = self.sip_ver(prob)
                # if slv_report.result == SolveResult.SATISFIED:
                    # VerificationProcess.logger.info(f'Verification problem {prob.id} is satisfied from bound analysis.')
                # else:
                # slv_report = self.lp_ver(prob)
                # if slv_report.result == SolveResult.SATISFIED:
                    # VerificationProcess.logger.info(f'Verification problem {prob.id} is satisfied from LP analysis.')
                # elif slv_report.result == SolveResult.UNSATISFIED:
                    # VerificationProcess.logger.info(f'Verification problem {prob.id} is unsatisfied from LP analysis.')
                # else:
                slv_report = self.complete_ver(prob)

                # VerificationProcess.logger.info('Subprocess {} '
                                                # 'finished job {}, '
                                                # 'result: {}, '
                                                # 'time: {:.2f}.'.format(self.id, 
                                                                       # prob.id, 
                                                                       # slv_report.result.value, 
                                                                       # slv_report.runtime))
                self.reporting_queue.put(slv_report)
            except queue.Empty:
                # VerificationProcess.logger.info(f"Subprocess {self.id} terminated because of empty job queue.")
                break


    def complete_ver(self, prob):
        slv = MILPSolver(prob, self.solver_params, self.sip_params)
        return  slv.solve()

    def sip_ver(self, prob):
        """
        Attempts to solve a verification problem using symbolic interval propagation.

        Returns:

            SolveReport
        """
        start = timer()
        if prob.satisfies_spec():
            return SolveReport(SolveResult.SATISFIED, timer() - start, None)
        else:
            return SolveReport(SolveResult.UNKNOWN, timer()-start, None)

    def lp_ver(self, prob):
        start = timer()
        slv = MILPSolver(prob, self.solver_params, self.sip_params, lp=True) 
        slv_report =  slv.solve()
        if slv_report.result == SolveResult.SATISFIED:
            return SolveReport(SolveResult.SATISFIED, timer() - start, None)
        elif slv_report.result == SolveResult.UNSATISFIED:
            nn_out = prob.nn.predict(slv_report.cex)
            if not prob.spec.is_satisfied(nn_out, nn_out):
                return SolveReport(SolveResult.UNSATISFIED, timer()-start, slv_report.cex)
        
        return SolveReport(SolveResult.UNKNOWN, timer()-start, None)

