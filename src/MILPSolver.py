# ************
# File: MILPSolver.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Solves a verification problem by tranlating it into MILP.
# ************

from gurobipy import *
from src.MILPEncoder import MILPEncoder
from src.IdealFormulation import IdealFormulation
from src.DepCuts import DepCuts
# from src.utils.Logger import get_logger
from src.utils.SolveResult import SolveResult
from src.utils.SolveReport import SolveReport
from src.utils.Activations import Activations
from src.utils.ReluState import ReluState
from timeit import default_timer as timer
import numpy as np

class MILPSolver:

    # logger = None

    def __init__(self, prob, params, sip_params, lp=False):
        """
        Arguments:

            prob: VerificationProblem.
            
            params: Params.solver 

            logfile: str of path to logfile.
        """
        MILPSolver.prob = prob
        MILPSolver.params = params
        MILPSolver.sip_params = sip_params
        MILPSolver.status = SolveResult.UNKNOWN
        MILPSolver.lp = lp
        # if MILPSolver.logger is None:
            # MILPSolver.logger = get_logger(__name__, params.logger.LOGFILE)

    def solve(self):
        """
        Builds and solves the MILP program of the verification problem.

        Returns:

            SolveReport
        """
        start = timer()
        # encode into milp
        me = MILPEncoder(MILPSolver.prob,
                         MILPSolver.params.logger.LOGFILE, 
                         MILPSolver.params.INTRA_DEP_CONSTRS,
                         MILPSolver.params.INTER_DEP_CONSTRS)
        if MILPSolver.lp == True:
            gmodel = me.lp_encode()
        else:
            gmodel = me.encode()
        # Set gurobi parameters
        pgo = 1 if MILPSolver.params.PRINT_GUROBI_OUTPUT  == True else 0
        gmodel.setParam('OUTPUT_FLAG', pgo)
        tl = MILPSolver.params.TIMEOUT
        if tl != -1 : gmodel.setParam('TIME_LIMIT', tl)
        if not MILPSolver.params.DEFAULT_CUTS: 
            MILPSolver.disable_default_cuts(gmodel)
        gmodel._vars = gmodel.getVars()
        # set callback cuts 
        MILPSolver.id_form = IdealFormulation(MILPSolver.prob,
                                              gmodel, 
                                              MILPSolver.params.IDEAL_FREQ,
                                              MILPSolver.params.logger.LOGFILE)
        MILPSolver.dep_cuts = DepCuts(MILPSolver.prob,
                                      gmodel,
                                      MILPSolver.params.DEP_FREQ,
                                      MILPSolver.params.INTRA_DEP_CUTS,
                                      MILPSolver.params.INTER_DEP_CUTS,
                                      MILPSolver.sip_params,
                                      MILPSolver.params.logger.LOGFILE)
        # Optimise
        if MILPSolver.params.callback_enabled() and MILPSolver.lp == False:
            gmodel.optimize(MILPSolver._callback)
        else:
            gmodel.optimize()

        runtime =  timer() - start
        cex = None  
        if MILPSolver.status == SolveResult.BRANCH_THRESHOLD:
            result = SolveResult.BRANCH_THRESHOLD
        elif gmodel.status == GRB.OPTIMAL:
            cex_shape = MILPSolver.prob.spec.input_layer.input_shape
            cex = np.zeros(cex_shape)
            for i in itertools.product(*[range(j) for j in cex_shape]):
                cex[i] = MILPSolver.prob.spec.input_layer.out_vars[i].x
            result = SolveResult.UNSATISFIED
        elif gmodel.status == GRB.TIME_LIMIT:
            result = SolveResult.TIMEOUT
        elif gmodel.status == GRB.INTERRUPTED:
            result = SolveResult.INTERRUPTED
        elif gmodel.status == GRB.INFEASIBLE or gmodel.status == GRB.INF_OR_UNBD:
            result = SolveResult.SATISFIED
        else:
            result = SolveResult.UNKNOWN
        
        # MILPSolver.logger.info('Verification problem {} solved, '
                               # 'LP: {}, '
                               # 'time: {:.2f}, '
                               # 'result: {}.'
                               # .format(MILPSolver.prob.id,
                                       # MILPSolver.lp,
                                       # runtime,
                                       # result.value))
        
        return SolveReport(result, runtime, cex)



    @staticmethod
    def _callback(model, where):
        """
        Gurobi callback function.
        """
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
                if MILPSolver.params.IDEAL_CUTS == True:
                    # MILPSolver.id_form.add_cuts()
                    pass
                if MILPSolver.params.dep_cuts_enabled():
                    MILPSolver.dep_cuts.add_cuts()
        elif MILPSolver.params.MONITOR_SPLIT==True and where==GRB.Callback.MIP:
            MILPSolver.monitor_milp_nodes(model)


    @staticmethod
    def monitor_milp_nodes(model): 
        """
        Monitors the number of MILP nodes solved. Terminates the MILP if the
        number exceeds the BRANCH_THRESHOLD.

        Arguments:

            model: Gurobi model.
        """
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
        if nodecnt > MILPSolver.params.BRANCH_THRESHOLD:
            MILPSolver.status = SolveResult.BRANCH_THRESHOLD
            model.terminate()


    @staticmethod
    def add_dep_cuts(model):
        """
        Adds dependency cuts.

        Arguments:

            model: Gurobi model.
        """
        ts = timer()
        # compute runtime bounds
        delta, _delta = MILPSolver._get_current_delta(model)
        MILPSolver.prob.set_bounds(_delta)
        # get linear desctiption of the current stabilised binary variables
        le = MILPSolver._get_lin_descr(delta, _delta)
        # build dependency graph and add dependencies
        dg = DependencyGraph(MILPSolver.prob.spec.input_layer,
                             MILPSolver.prob.nn, 
                             MILPSolver.logfile, 
                             MILPSolver.params.INTRA_DEP_CONSTRS, 
                             MILPSolver.params.INTER_DEP_CONSTRS)
        for lhs_node in dg.nodes:
            for rhs_node, dep in dg.nodes[lhs_node].adjacent:
                # get the nodes in the dependency
                l1 = node1.layer 
                n1 = node1.index
                delta1 = MILPSolver.prob.nn.layers[l1].delta_vars[n1]
                l2 = node2.layer 
                n2 = node2.index
                delta2 = MILPSolver.prob.nn.layers[l2].delta_vars[n2]
                # add the constraint as per the type of the dependency
                if dep == DependencyType.INACTIVE_INACTIVE:
                    model.cbCut(delta2 <= le + delta1)
                elif dep == DependencyType.INACTIVE_ACTIVE:
                    model.cbCut(1 - delta2 <= le + delta1)
                elif dep == DependencyType.ACTIVE_INACTIVE:
                    model.cbCut(delta2 <= le + 1 - delta1)
                elif dep == DependencyType.ACTIVE_ACTIVE:
                    model.cbCut(1 - delta2 <= le + 1 - delta1)
        te = timer()
        # MILPSolver.logger.info('Added dependency cuts, #cuts: {dg.get_total_depts_count()}, time: {te - ts}')

    @staticmethod
    def _get_current_delta(model):
        """
        Fetches the binary variables and their current values.

        Arguments:

            model: Gurobi model.

        Returns:
            
            list of all binary variables, list of their current values.
        """
        delta = []
        _delta = []
        for i in MILPSolver.prob.nn.layers:
            (s, e) = vmodel.get_var_indices(i.depth, 'delta')
            d = model._vars[s:e]
            _d = np.asarray(model.cbGetNodeRel(d))
            delta.append(d)
            _delta.append(_d)

        return delta, _delta

    def _get_lin_descr(delta, _delta):
        """
        Creates a linear expression of current integral binary variables.
        Current dependency cuts are only sound when this expression is
        satisfied so the cuts are added in conjuction with the expression.

        Arguments:

            delta: list of binary variables.

            _delta: list of values of delta.

        Returns:
            
            LinExpr
        """

        le = LinExpr()
        for i in range(len(vmodel.lmodel.layers)):
            l = vmodel.lmodel.layers[i]
            if l.activation == Activations.relu:
                d = delta[i]
                _d = _delta[i]
                for j in range(len(d)):
                    if _d[j] == 0 and not vmodel.lmodel.layers[i].is_stable(j):
                        le.addTerms(1, d[j])
                    elif _d[j] == 1 and not vmodel.lmodel.layers[i].is_stable(j):
                        le.addConstant(1)
                        le.addTerms(-1, d[j])
        
        return le

    


    @staticmethod
    def disable_default_cuts(gmodel):
        """
        Disables Gurobi default cuts.

        Arguments:

            gmodel: Gurobi Model.

        Returns:

            None.
        """
        gmodel.setParam('PreCrush', 1)
        gmodel.setParam(GRB.Param.CoverCuts,0)
        gmodel.setParam(GRB.Param.CliqueCuts,0)
        gmodel.setParam(GRB.Param.FlowCoverCuts,0)
        gmodel.setParam(GRB.Param.FlowPathCuts,0)
        gmodel.setParam(GRB.Param.GUBCoverCuts,0)
        gmodel.setParam(GRB.Param.ImpliedCuts,0)
        gmodel.setParam(GRB.Param.InfProofCuts,0)
        gmodel.setParam(GRB.Param.MIPSepCuts,0)
        gmodel.setParam(GRB.Param.MIRCuts,0)
        gmodel.setParam(GRB.Param.ModKCuts,0)
        gmodel.setParam(GRB.Param.NetworkCuts,0)
        gmodel.setParam(GRB.Param.ProjImpliedCuts,0)
        gmodel.setParam(GRB.Param.StrongCGCuts,0)
        gmodel.setParam(GRB.Param.SubMIPCuts,0)
        gmodel.setParam(GRB.Param.ZeroHalfCuts,0)
        gmodel.setParam(GRB.Param.GomoryPasses,0)


