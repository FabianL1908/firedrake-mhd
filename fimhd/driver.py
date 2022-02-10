from firedrake import *
import argparse

def get_default_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseN", type=int, default=20)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--nref", type=int, default=2)
    parser.add_argument("--Re", nargs='+', type=float, default=[1])
    parser.add_argument("--Rem", nargs='+', type=float, default=[1])
    parser.add_argument("--gamma", type=float, default=10000)
    parser.add_argument("--advect", type=float, default=1)
    parser.add_argument("--S", nargs='+', type=float, default=[1])
    parser.add_argument("--hierarchy", choices=["bary", "uniform"], default="bary")
    parser.add_argument("--discr", choices=["rt", "bdm", "cg"], required=True)
    parser.add_argument("--solver-type", choices=list(solvers.keys()), default="lu")
    parser.add_argument("--testproblem", choices=["ldc", "hartmann", "Wathen", "hartmann2"], default="Wathen")
    parser.add_argument("--linearisation", choices=["picard", "mdp", "newton"], required=True)
    parser.add_argument("--stab", default=False, action="store_true")
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument("--output", default=False, action="store_true")

    return parser
