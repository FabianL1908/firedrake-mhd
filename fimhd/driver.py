from firedrake import *
import argparse
from fimhd.utils import message

def get_default_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseN", type=int, default=10)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2)
    parser.add_argument("--nref", type=int, default=1)
    parser.add_argument("--Re", nargs='+', type=float, default=[1])
    parser.add_argument("--Rem", nargs='+', type=float, default=[1])
    parser.add_argument("--gamma", type=float, default=10000)
    parser.add_argument("--advect", type=float, default=1)
    parser.add_argument("--S", nargs='+', type=float, default=[1])
    parser.add_argument("--hierarchy", choices=["bary", "uniform"], default="bary")
    parser.add_argument("--ns-discr", choices=["hdivrt", "hdivbdm", "sv", "th"], required=True)
    parser.add_argument("--mw-discr", choices=["BE", "Br"], required=True)
    parser.add_argument("--mhd-type", choices=["standard", "hall", "boussinesq"], required=True)
#    parser.add_argument("--solver-type", choices=list(solvers.keys()), default="lu")
    parser.add_argument("--solver-type", type=str, default="lu")
    parser.add_argument("--linearisation", choices=["picard", "mdp", "newton"], required=True)
    parser.add_argument("--stab", default=False, action="store_true")
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument("--output", default=False, action="store_true")
    parser.add_argument("--tinyasm", default=False, action="store_true")

    return parser

def run_solver(solver, res, rems, ss):
    for rem in rems:
        for s in ss:
            for re in res:
#                try:
                    solver.solve(re, rem, s)
                # If solve fails report 0 as iteration number
#                except Exception as e:
#                    message(e, solver.problem.mesh)
