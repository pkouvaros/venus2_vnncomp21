import sys
import multiprocessing
import argparse
import numpy as np
from timeit import default_timer as timer
import datetime
from alive_progress import alive_bar

#### to stop seeing various warning and system messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')
# sys.stderr = stderr

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
####

from src.Specification import Specification
from src.AdversarialRobustness import AdversarialRobustness
from src.Layers import Input
from src.input.AcasProp import acas_properties, acas_denormalise_input, acas_normalise_input
from src.input.InputLoader import InputLoader
from src.input.VNNLIBParser import VNNLIBParser
from src.Formula import NegationFormula
from src.Parameters import  Params
from src.NeuralNetwork import NeuralNetwork 
from src.Verifier import Verifier
from src.utils.SplitStrategy import SplitStrategy
from src.utils.SolveResult import SolveResult
from src.utils.ReluApproximation import ReluApproximation
from src.utils.OSIPMode import OSIPMode
from src.utils.Logger import get_logger

verifier_result_to_string = {"True": "Satisfied", True: "Satisfied", "False": "NOT Satisfied", False: "NOT Satisfied", "Interrupted": "Interrupt", "Timeout": "Timed Out", "Unknown": "Undecided"}

def fc_params(options, model):
    params = Params()
    params.logger.LOGFILE = options.logfile
    params.solver.logger.LOGFILE = options.logfile
    params.splitter.logger.LOGFILE = options.logfile
    params.solver.TIME_LIMIT = int(options.timeout)
    params.solver.INTRA_DEP_CUTS = False
    params.sip.OSIP_CONV = OSIPMode.OFF
    params.sip.OSIP_FC = OSIPMode.OFF
    params.verifier.VER_PROC_NUM = options.workers
    params.splitter.SPLIT_PROC_NUM = 0
    params.splitter.INTER_DEPS = True
    params.sip.RELU_APPROXIMATION = ReluApproximation.VENUS_HEURISTIC
    relus = model.get_n_relu_nodes()

    if model.layers[0].input_size < 10:
        params.solver.INTER_DEP_CONSTRS = True
        params.solver.IDEAL_CUTS = False
        params.solver.INTRA_DEP_CONSTRS = False
        params.solver.INTER_DEP_CUTS = False
        params.solver.MONITOR_SPLIT = False
        params.splitter.STABILITY_RATIO_CUTOFF = 0.75
        params.splitter.SPLIT_STRATEGY = SplitStrategy.INPUT
        if relus < 1000:
            params.splitter.SPLIT_PROC_NUM = 2
            params.solver.INTER_DEP_CONSTRS = False
    else:   
        params.solver.IDEAL_CUTS = True
        params.solver.INTER_DEP_CONSTRS = True
        params.solver.INTRA_DEP_CONSTRS = True
        params.solver.INTER_DEP_CUTS = True
        params.solver.MONITOR_SPLIT = True
        params.splitter.SPLIT_STRATEGY = SplitStrategy.NODE
        if relus < 1000:
            params.splitter.BRANCHING_DEPTH = 2
            params.solver.BRANCH_THRESHOLD = 10000
        elif relus < 2000:
            params.splitter.BRANCHING_DEPTH = 2
            params.solver.BRANCH_THRESHOLD = 5000
        else:
            params.splitter.BRANCHING_DEPTH = 7
            params.solver.BRANCH_THRESHOLD = 300

    return params

def conv_params(options, model):
    params = Params()
    params.logger.LOGFILE = options.logfile
    params.solver.logger.LOGFILE = options.logfile
    params.splitter.logger.LOGFILE = options.logfile
    params.solver.TIME_LIMIT = int(options.timeout)
    params.solver.INTRA_DEP_CUTS = False
    params.solver.IDEAL_CUTS = True
    params.sip.OSIP_CONV = OSIPMode.OFF
    params.sip.OSIP_FC = OSIPMode.OFF
    params.verifier.VER_PROC_NUM = options.workers
    params.splitter.SPLIT_PROC_NUM = 0
    params.splitter.INTER_DEPS = True
    params.splitter.STABILITY_RATIO_CUTOFF = 0.9
    params.solver.INTER_DEP_CONSTRS = True
    relus = model.get_n_relu_nodes()

    if relus <= 10000 and len(model.layers) <=5:
        params.sip.RELU_APPROXIMATION = ReluApproximation.VENUS_HEURISTIC
    else:
        params.sip.RELU_APPROXIMATION = ReluApproximation.MIN_AREA


    if relus <= 4000:
        params.solver.INTRA_DEP_CONSTRS = True
        params.solver.INTER_DEP_CUTS = True
    else:
        params.solver.INTRA_DEP_CONSTRS = False
        params.solver.INTER_DEP_CUTS = False
    if relus <= 10000:
        params.splitter.SPLIT_STRATEGY = SplitStrategy.NODE
        params.solver.MONITOR_SPLIT = True
        params.splitter.BRANCHING_DEPTH = 2
        params.solver.BRANCH_THRESHOLD = 50
    else:
        params.solver.MONITOR_SPLIT = False
        params.splitter.SPLIT_STRATEGY = SplitStrategy.NONE

    return params



def parameters(options, model):
    if model.is_fc():
        return fc_params(options, model)
    else:
        return conv_params(options, model)


def verify_acas_property(options):
    prop = acas_properties[options.acas_prop]
    input_bounds = (np.array(prop['bounds']['lower']), np.array(prop['bounds']['upper']))
    spec = GenSpec(input_bounds, prop['output'])
    encoder_params, splitting_params = params(options)
    verifier = VenusVerifier(options.net, spec, encoder_params, splitting_params, options.print)
    start = timer()
    result,_,_,_,ctx = verifier.verify()
    end = timer()
    runtime = end - start
    str_result = verifier_result_to_string[result]
    print("{} over {}".format(prop['name'], options.net), "is", str_result, "in {:9.4f}s".format(runtime))
    if result == "True":
        nmodel = keras.models.load_model(options.net)
        denormalised_ctx = acas_denormalise_input(ctx).reshape(1,-1)
        network_output = nmodel.predict(x=denormalised_ctx, batch_size=1)
        print("\t\tCounter-example:", list(ctx))
        print("\t\tDenormalised   :", list(denormalised_ctx))
        print("\t\tNetwork output :", list(network_output[0]))
    print("")

    return result, runtime

def verify_adversarial_robustness(options):
    #summary info
    sat = 0 
    unsat = 0
    total_time = 0
    total_sat_time = 0
    total_unsat_time = 0
    # setup parameters
    params = parameters(options)
    #load model
    model = NeuralNetwork(options.net, options.logfile)
    model.load()
    num_classes = model.layers[-1].output_size
    # load images 
    input_loader = InputLoader(options.lrob_input, 
                               normalise=True, 
                               min_value=options.min_input_value, 
                               max_value=options.max_input_value, 
                               shape=model.layers[0].input_shape)
    imgs = input_loader.load()
    print(f'\nModel   : {options.net}')
    print(f'Property: Adversarial Robustness ({options.lrob_radius})')
    print(f'Input   : {options.lrob_input}\n')
    with alive_bar(len(imgs),bar='blocks',spinner='classic') as bar:
        for k in imgs:
            label = model.classify(imgs[k], options.mean, options.std)
            # create specification
            spec = AdversarialRobustness(imgs[k],
                                         label, 
                                         num_classes, 
                                         options.lrob_radius, 
                                         options.mean, 
                                         options.std, 
                                         0, 
                                         1,
                                         str(k))
            # create verifier
            verifier = Verifier(model, spec, params)
            ver_report = verifier.verify(options.complete)
            bar()
            if ver_report.result == SolveResult.SATISFIED:
                sat += 1
                total_sat_time += ver_report.runtime
            elif ver_report.result == SolveResult.UNSATISFIED:
                unsat += 1
                total_unsat_time += ver_report.runtime
            total_time += ver_report.runtime

            with open(options.sumfile, 'a') as f:
                f.write('{:<12}{:6.2f}\n'.format(ver_report.result.value, ver_report.runtime))

    avg_sat = 0 if sat == 0 else total_sat_time / sat
    avg_unsat = 0 if unsat == 0 else total_unsat_time / unsat

    with open(options.sumfile, 'a') as f:
        f.write('\n\nVerified: {}\tSATisfied: {}\tUNSATisfied: {}\tTimeouts: {}\n\n'.format(sat + unsat, sat, unsat, len(imgs) - sat - unsat))
        f.write('Total Time:       {:6.2f}\tAvg Time:       {:6.2f}\n'.format(total_time, total_time / len(imgs)))
        f.write('Total SAT Time:   {:6.2f}\tAvg SAT Time:   {:6.2f}\n'.format(total_sat_time, avg_sat))
        f.write('Total UNSAT Time: {:6.2f}\tAvg UNSAT Time: {:6.2f}\n\n'.format(total_unsat_time, avg_unsat))

    print('\nVerified: {}\tSATisfied: {}\tUNSATisfied: {}\tTimeouts: {}\n'.format(sat + unsat, sat, unsat, len(imgs) - sat - unsat))
    print('Total Time:       {:6.2f}\tAvg Time:       {:6.2f}'.format(total_time, total_time / len(imgs)))
    print('Total SAT Time:   {:6.2f}\tAvg SAT Time:   {:6.2f}'.format(total_sat_time, avg_sat))
    print('Total UNSAT Time: {:6.2f}\tAvg UNSAT Time: {:6.2f}'.format(total_unsat_time, avg_unsat))


def verify_spec(model, params, i_b, o_f, options):
    i_b = [i_b[0].reshape(model.layers[0].input_shape), i_b[1].reshape(model.layers[0].input_shape)]
    f = NegationFormula(o_f).to_NNF()
    input_layer = Input(i_b[0], i_b[1])
    # print(model.mean, model.std)
    # import sys
    # sys.exit()
    input_layer.post_bounds.normalise(model.mean * -1, model.std)
    input_layer.pre_bounds = input_layer.post_bounds
    spec = Specification(Input(i_b[0], i_b[1]), f, options.property)
    # create verifier
    verifier = Verifier(model, spec, params)
    ver_report = verifier.verify(options.complete)

    return ver_report

def verify(options):
    time_elapsed = 0
    # load model
    model = NeuralNetwork(options.net, options.logfile)
    model.load()
    # setup parameters
    params = parameters(options, model)
    # load specification
    vnn_parser = VNNLIBParser(options.property, 
                              model.layers[0].input_size, 
                              model.layers[-1].output_size)
    i_b, o_f, i_cl = vnn_parser.parse() 
    if len(i_cl) == 0:
        ver_report = verify_spec(model, params, i_b, o_f, options)
        with open('testing.txt','a') as f:
            f.write(f'{ver_report.runtime}\n')
        return ver_report.result
    else:
        for spec in i_cl:
            if not o_f is None and not spec[1] is None:
                f = ConjFormula(o_f, spec[1])
            elif not o_f is None and spec[1] is None:
                f = o_f
            elif o_f is None and not spec[1] is None:
                f = spec[1]
            else:
                raise Exception('Not output constraints')
            ver_report = verify_spec(model, params, spec[0], f, options)
            if ver_report.result == SolveResult.UNSATISFIED:
                return SolveResult.UNSATISFIED
            else:
                time_left = params.solver.TIME_LIMIT - ver_report.runtime
                if time_left <= 0:
                    return SolveResult.TIMEOUT
                else:
                    params.solver.TIME_LIMIT =  time_left

        return SolveResult.SATISFIED


def boolean_string(s):
    assert  s in ['False', 'True']
    return s == 'True'

def array_string(s):
    return np.array([float(x) for x in s.split(',')])

def main():
    parser = argparse.ArgumentParser(description="Venus Example")
    parser.add_argument("--property", 
                        type=str,
                        required=True, 
                        help="Verification property, one of acas or lrob or vnnlib file.")
    parser.add_argument("--net", 
                        type=str, 
                        required=True, 
                        help="Path to the neural network in ONNX or Keras format.")
    parser.add_argument("--acas_prop", 
                        type=int, 
                        default=None, 
                        help="Acas property number from 0 to 10. Default value is 1.")
    parser.add_argument("--lrob_input", 
                        type=str, 
                        default=None, 
                        help="Path to the input and label.")
    parser.add_argument("--lrob_radius", 
                        default=0.1, 
                        type=float, 
                        help="Perturbation radius for L_inifinity norm. Default value is 0.1.")
    parser.add_argument("--mean", 
                        default=0,
                        type=array_string, 
                        help="Normalisation mean.")
    parser.add_argument("--std", 
                        default=1, 
                        type=array_string, 
                        help="Normalisation std.")
    parser.add_argument("--min_input_value", 
                        default=0, 
                        type=float,
                        help="Minimum valid value of the input nodes")
    parser.add_argument("--max_input_value", 
                        default=255,
                        type=float,
                        help="Maximum valid value of the input nodes")
    parser.add_argument("--st_ratio", 
                        default=0.5, 
                        type=float, 
                        help="Cutoff value of the stable ratio during the splitting procedure. Default value is 0.5.")
    parser.add_argument("--depth_power", 
                        default=1.0, 
                        type=float, 
                        help="Parameter for the splitting depth. Higher values favour splitting. Default value is 1.")
    parser.add_argument("--splitters", 
                        default=0, 
                        type=int, 
                        help="Determines the number of splitting processes = 2^splitters. Default value is 0. -1 for None.")
    parser.add_argument("--workers", 
                        default=multiprocessing.cpu_count(), 
                        type=int, 
                        help="Number of worker processes. Default value is 1.")
    parser.add_argument("--intra_offline_deps", 
                        default=True, 
                        type=boolean_string, 
                        help="Whether to include offline intra dependency contrainsts (before starting the solver) or not. Default value is True.")
    parser.add_argument("--inter_offline_deps", 
                        default=True, 
                        type=boolean_string,
                        help="Whether to include offline inter dependency contrainsts (before starting the solver) or not. Default value is True.")
    parser.add_argument("--intra_online_deps", 
                        default=True, 
                        type=boolean_string, 
                        help="Whether to include online intra dependency cuts (through solver callbacks) or not. Default value is True.")
    parser.add_argument("--inter_online_deps", 
                        default=True, 
                        type=boolean_string,
                        help="Whether to include online inter dependency cuts (through solver callbacks) or not. Default value is True.")
    parser.add_argument("--ideal_cuts", 
                        default=True, 
                        type=boolean_string, 
                        help="Whether to include online ideal cuts (through solver callbacks) or not. Default value is True.")
    parser.add_argument("--split_strategy", 
                        choices=["node","nodeonce","input","inputnode","inputnodeonce","nodeinput","nodeonceinput","inputnodealt","inputnodeoncealt","none"], 
                        default="node", help="Strategies for diving the verification problem")
    parser.add_argument("--monitor_split",
                        default=False, 
                        type=boolean_string,  
                        help="If true branching is initiated only after the <branch_threshold> of MILP nodes is reached")
    parser.add_argument("--branching_depth", 
                        default=7, 
                        type=int, 
                        help="Maximum branching depth")
    parser.add_argument("--branch_threshold", 
                        default=5000, 
                        type=int, 
                        help="MILP node thresholf before inititing branching")
    parser.add_argument("--timeout", 
                        default=3600, 
                        type=float, 
                        help="Timeout in seconds. Default value is 3600.")
    parser.add_argument("--logfile", 
                        default="venus_log_" + str(datetime.datetime.now()) + ".txt",
                        type=str, 
                        help="Path to logging file.")
    parser.add_argument("--sumfile", 
                        default="summary" + ".txt",
                        type=str, 
                        help="Path to summary file.")
    parser.add_argument("--print", 
                        default=False, 
                        type=boolean_string, 
                        help="Print extra information or not. Default value is False.")
    parser.add_argument("--complete",
                        default=True, 
                        type=boolean_string, 
                        help="Complete or incomplete verification")
    parser.add_argument("--osip_conv",
                        default='split', 
                        type=str,
                        help="OSIP mode of operation for convolutional layers, one of 'on', 'off', 'node_once', 'node_always'")
    parser.add_argument("--osip_conv_nodes",
                        default=200, 
                        type=int,
                        help="Number of optimised nodes during OSIP for convolutional layers")
    parser.add_argument("--osip_fc",
                        default='split', 
                        type=str, 
                        help="OSIP mode of operation for fully connected layers, one of 'on', 'off', 'node_once', 'node_always'")
    parser.add_argument("--osip_fc_nodes",
                        default=3, 
                        type=int,
                        help="Number of optimised nodes during OSIP for fully connected layers")
    parser.add_argument("--osip_timelimit",
                        default=200, 
                        type=int,
                        help="Timelimit in seconds for OSIP")
    parser.add_argument("--relu_approx",
                        default='min_area', 
                        type=str,
                        help="Relu approximation: 'min_area' or 'identity' or 'venus' or 'parallel' or 'zero'")
    ARGS = parser.parse_args()


    if ARGS.property == 'acas':
        result, runtime = verify_acas_property(ARGS)
    elif ARGS.property == 'lrob':
        verify_adversarial_robustness(ARGS)
    elif os.path.exists(ARGS.property):
        res = verify(ARGS)

    with open(ARGS.sumfile,'w') as f:
        if res == SolveResult.SATISFIED:
            f.write('holds\n')
        elif res == SolveResult.UNSATISFIED:
            f.write('violated\n')
        elif res == SolveResult.TIMEOUT:
            f.write('timeout\n')
        else:
            f.write('unknown\n')

if __name__ == "__main__":
    main()
