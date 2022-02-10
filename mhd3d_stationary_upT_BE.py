# -*- coding: utf-8 -*-
from firedrake import *

import alfi
from alfi.stabilisation import *
from alfi.transfer import *

from pyop2.datatypes import IntType
import ufl.algorithms

from datetime import datetime
import argparse
import numpy
import sys
import os

import petsc4py
petsc4py.PETSc.Sys.popErrorHandler()

# Parallel distribution parameters
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

# Definition of Burman Stabilisation term to avoid oscillations in velocity u
def BurmanStab(B, C, wind, stab_weight, mesh):
    n = FacetNormal(mesh)
    h = FacetArea(mesh)
    beta = avg(facet_avg(sqrt(inner(wind, wind)+1e-10)))
    gamma1 = stab_weight  # as chosen in doi:10.1016/j.apnum.2007.11.001
    stabilisation_form = 0.5 * gamma1 * avg(h)**2 * beta * dot(jump(grad(B), n), jump(grad(C), n))*dS
    return stabilisation_form


# Definition of outer Schur complement for the order ((E,B),(u,p)), which is Navier-Stokes block + Lorentz force
class SchurPC(AuxiliaryOperatorPC):
    def form(self, pc, V, U):
        mesh = V.function_space().mesh()
        [u, p] = split(U)
        [v, q] = split(V)
        Z = V.function_space()
        U_ = Function(Z)

        state = self.get_appctx(pc)['state']
        [u_n, p_n, B_n, E_n] = split(state)

        eps = lambda x: sym(grad(x))

        # Weak form of NS + Lorentz force
        A = (
                2/Re * inner(eps(u), eps(v))*dx
              + gamma * inner(div(u), div(v)) * dx
              - inner(p, div(v)) * dx
              - inner(div(u), q) * dx
              + S * inner(cross(B_n, cross(u, B_n)), v) * dx
             )

        # For H(div)-L2 discretization we have to add a DG-formulation of the advection and diffusion term
        if discr in ["rt", "bdm"]:
            h = CellVolume(mesh)/FacetArea(mesh)
            uflux_int = 0.5*(dot(u, n) + theta*abs(dot(u, n)))*u
            uflux_ext_1 = 0.5*(inner(u, n) + theta*abs(inner(u, n)))*u
            uflux_ext_2 = 0.5*(inner(u, n) - theta*abs(inner(u, n)))*u_ex

            A_DG = (
                 - 1/Re * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS
                 - 1/Re * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS
                 + 1/Re * sigma/avg(h) * inner(2*avg(outer(u, n)), 2*avg(outer(v, n))) * dS
                 - inner(outer(v, n), 2/Re*sym(grad(u))) * ds
                 - inner(outer(u-u_ex, n), 2/Re*sym(grad(v))) * ds(bcs_ids_apply)
                 + 1/Re*(sigma/h)*inner(v, u-u_ex) * ds(bcs_ids_apply)
                 - advect * dot(u, div(outer(v, u))) * dx
                 + advect * dot(v('+')-v('-'), uflux_int('+')-uflux_int('-'))*dS
                 + advect * dot(v, uflux_ext_1) * ds
                 + advect * dot(v, uflux_ext_2) * ds(bcs_ids_apply)
            )

            if bcs_ids_dont_apply is not None:
                A_DG += (
                    - inner(outer(u, n), 2/Re*sym(grad(v))) * ds(bcs_ids_dont_apply)
                    + 1/Re*(sigma/h)*inner(v, u) * ds(bcs_ids_dont_apply)
                   )

            # Linearize A_DG
            A_DG = action(A_DG, U_)
            A_DG_linear = derivative(A_DG, U_, U)
            A_DG_linear = replace(A_DG_linear, {split(U_)[0]: u_n, split(U_)[1]: p_n})
            A = A + A_DG_linear
        elif discr in ["cg"]:
            A = A + advect * inner(dot(grad(u_n), u), v) * dx + advect * inner(dot(grad(u), u_n), v) * dx

        if stab:
            stabilisation2 = BurmanStabilisation(V.function_space().sub(0), state=z_last_u, h=FacetArea(mesh), weight=stab_weight)
            stabilisation_form_2 = stabilisation2.form(u, v)
            A = A + advect * stabilisation_form_2

        bcs = [DirichletBC(V.function_space().sub(0), 0, "on_boundary"),
               PressureFixBC(V.function_space().sub(1), 0, 1),
               ]

        return (A, bcs)

# Definition of outer Schur complement for the order ((u,p),(B,E))
class SchurPCBE(AuxiliaryOperatorPC):
    def form(self, pc, V, U):
        [B, E] = split(U)
        [C, Ff] = split(V)
        state = self.get_appctx(pc)['state']
        [u_n, p_n, T_n, B_n, E_n] = split(state)

        A = (
             + inner(E, Ff) * dx
             + inner(cross(u_n, B), Ff) * dx
             - 1/Pm * inner(B, curl(Ff)) * dx
             + inner(curl(E), C) * dx
             + 1/Pm * inner(div(B), div(C)) * dx
             + gamma2 * inner(div(B), div(C)) * dx
        )

        bcs = [DirichletBC(V.function_space().sub(0), 0, "on_boundary"),
               DirichletBC(V.function_space().sub(1), 0, "on_boundary"),
               ]

        return (A, bcs)


# We fix the pressure at one vertex on every level
class PressureFixBC(DirichletBC):
    def __init__(self, V, val, subdomain, method="topological"):
        super().__init__(V, val, subdomain, method)
        sec = V.dm.getDefaultSection()
        dm = V.mesh().topology_dm

        coordsSection = dm.getCoordinateSection()
        dim = dm.getCoordinateDim()
        coordsVec = dm.getCoordinatesLocal()

        (vStart, vEnd) = dm.getDepthStratum(0)
        indices = []
        for pt in range(vStart, vEnd):
            x = dm.getVecClosure(coordsSection, coordsVec, pt).reshape(-1, dim).mean(axis=0)
            if x.dot(x) == 0.0:  # fix [0, 0] in original mesh coordinates (bottom left corner)
                if dm.getLabelValue("pyop2_ghost", pt) == -1:
                    indices = [pt]
                break

        nodes = []
        for i in indices:
            if sec.getDof(i) > 0:
                nodes.append(sec.getOffset(i))

        if V.mesh().comm.rank == 0:
            nodes = [0]
        else:
            nodes = []
        self.nodes = numpy.asarray(nodes, dtype=IntType)

        if len(self.nodes) > 0:
            print("Fixing nodes %s" % self.nodes)
        import sys
        sys.stdout.flush()


# Definition of different solver parameters

# Increase buffer size for MUMPS solver
ICNTL_14 = 5000

# LU solver
lu = {
    "snes_type": "newtonls",
    "snes_monitor": None,
    "snes_atol": 1.0e-8,
    "snes_rtol": 1.0e-13,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1,
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": ICNTL_14,
}

nsfsstar = {
    "ksp_type": "fgmres",
    "ksp_atol": 1.0e-7,
    "ksp_rtol": 1.0e-7,
    "ksp_max_it": 2,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_0_fields": "0,2",
    "pc_fieldsplit_1_fields": "1",
    "pc_fieldsplit_schur_factorization_type": "upper",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_0": {
         "ksp_type": "gmres",
         "ksp_max_it": 1,
         "ksp_norm_type": "unpreconditioned",
         "pc_type": "mg",
         "pc_mg_cycle_type": "v",
         "pc_mg_type": "full",
         "mg_levels_ksp_type": "fgmres",
         "mg_levels_ksp_convergence_test": "skip",
         "mg_levels_ksp_max_it": 6,
         "mg_levels_ksp_norm_type": "unpreconditioned",
         "mg_levels_pc_type": "python",
         "mg_levels_pc_python_type": "firedrake.ASMStarPC",
         "mg_levels_pc_star_backend": "tinyasm",
         # "mg_levels_pc_star_construct_dim": 0,
         # "mg_levels_pc_star_sub_sub_ksp_type": "preonly",
         # "mg_levels_pc_star_sub_sub_pc_type": "lu",
         # "mg_levels_pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack",
         "mg_coarse_ksp_type": "richardson",
         "mg_coarse_ksp_max_it": 1,
         "mg_coarse_ksp_norm_type": "unpreconditioned",
         "mg_coarse_pc_type": "python",
         "mg_coarse_pc_python_type": "firedrake.AssembledPC",
         "mg_coarse_assembled": {
             "mat_type": "aij",
             "pc_type": "telescope",
             "pc_telescope_reduction_factor": 1,
             "pc_telescope_subcomm_type": "contiguous",
             "telescope_pc_type": "lu",
             "telescope_pc_factor_mat_solver_type": "mumps",
             "telescope_pc_factor_mat_mumps_icntl_14": ICNTL_14,
         }
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "alfi.solver.DGMassInv"
    },
}

# Monolithic macrostar solver for (E,B)-block
nsfsmacrostar = {
    "ksp_type": "fgmres",
    "ksp_atol": 1.0e-7,
    "ksp_rtol": 1.0e-7,
    "ksp_max_it": 2,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_0_fields": "0,2",
    "pc_fieldsplit_1_fields": "1",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_0": {
           "ksp_type": "gmres",
           "ksp_max_it": 1,
           "ksp_norm_type": "unpreconditioned",
           "pc_type": "mg",
           "pc_mg_cycle_type": "v",
           "pc_mg_type": "full",
           "mg_levels_ksp_type": "fgmres",
           "mg_levels_ksp_convergence_test": "skip",
           "mg_levels_ksp_max_it": 6,
           "mg_levels_ksp_norm_type": "unpreconditioned",
           "mg_levels_pc_type": "python",
           "mg_levels_pc_python_type": "firedrake.PatchPC",
           "mg_levels_patch_pc_patch_save_operators": True,
           "mg_levels_patch_pc_patch_partition_of_unity": False,
           "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
           "mg_levels_patch_pc_patch_construct_dim": 0,
           "mg_levels_patch_pc_patch_construct_type": "python",
           "mg_levels_patch_pc_patch_construct_python_type": "alfi.MacroStar",
           "mg_levels_patch_sub_ksp_type": "preonly",
           "mg_levels_patch_sub_pc_type": "lu",
           "mg_levels_patch_sub_pc_factor_mat_solver_type": "umfpack",
           "mg_coarse_ksp_type": "richardson",
           "mg_coarse_ksp_max_it": 1,
           "mg_coarse_ksp_norm_type": "unpreconditioned",
           "mg_coarse_pc_type": "python",
           "mg_coarse_pc_python_type": "firedrake.AssembledPC",
           "mg_coarse_assembled": {
                 "mat_type": "aij",
                 "pc_type": "telescope",
                 "pc_telescope_reduction_factor": 1,
                 "pc_telescope_subcomm_type": "contiguous",
                 "telescope_pc_type": "lu",
                 "telescope_pc_factor_mat_solver_type": "mumps",
                 "telescope_pc_factor_mat_mumps_icntl_14": ICNTL_14,
             }
    },
    "fieldsplit_1": {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "alfi.solver.DGMassInv"
       },
}


# Fieldsplit solver for outer Schur complement with monolithic star solver
outerschurstar = {
    "ksp_type": "fgmres",
    "ksp_atol": 1.0e-7,
    "ksp_rtol": 1.0e-7,
    "pc_type": "python",
    "pc_python_type": "__main__.SchurPCBE",
    "aux_mg_transfer_manager": __name__ + ".transfermanager",
    "ksp_max_it": 2,
    "aux": {
         "ksp_type": "gmres",
         "ksp_max_it": 1,
         "ksp_norm_type": "unpreconditioned",
         "pc_type": "mg",
         "pc_mg_cycle_type": "v",
         "pc_mg_type": "full",
         "mg_levels_ksp_type": "fgmres",
         "mg_levels_ksp_convergence_test": "skip",
         "mg_levels_ksp_max_it": 6,
         "mg_levels_ksp_norm_type": "unpreconditioned",
         "mg_levels_pc_type": "python",
         "mg_levels_pc_python_type": "firedrake.ASMStarPC",
         "mg_levels_pc_star_backend": "tinyasm",
#         "mg_levels_pc_python_type": "firedrake.PatchPC",
#         "mg_levels_patch_pc_patch_save_operators": True,
#         "mg_levels_patch_pc_patch_partition_of_unity": False,
#         "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
#         "mg_levels_patch_pc_patch_construct_dim": 0,
#         "mg_levels_patch_pc_patch_construct_type": "star",
#         "mg_levels_patch_sub_ksp_type": "preonly",
#         "mg_levels_patch_sub_pc_type": "lu",
#         "mg_levels_patch_sub_pc_factor_mat_solver_type": "umfpack",
         "mg_coarse_ksp_type": "richardson",
         "mg_coarse_ksp_max_it": 1,
         "mg_coarse_ksp_norm_type": "unpreconditioned",
         "mg_coarse_pc_type": "python",
         "mg_coarse_pc_python_type": "firedrake.AssembledPC",
         "mg_coarse_assembled": {
               "mat_type": "aij",
               "pc_type": "telescope",
               "pc_telescope_reduction_factor": 1,
               "pc_telescope_subcomm_type": "contiguous",
               "telescope_pc_type": "lu",
               "telescope_pc_factor_mat_solver_type": "mumps",
               "telescope_pc_factor_mat_mumps_icntl_14": ICNTL_14,
          }
    },
}

# Fieldsplit solver for outer Schur complement with monolithic macrostar solver
outerschurmacrostar = {
    "ksp_type": "fgmres",
    "ksp_atol": 1.0e-7,
    "ksp_rtol": 1.0e-7,
    "pc_type": "python",
    "pc_python_type": "__main__.SchurPCBE",
    "aux_mg_transfer_manager": __name__ + ".transfermanager",
    "ksp_max_it": 2,
    "aux": {
           "ksp_type": "gmres",
           "ksp_max_it": 1,
           "ksp_norm_type": "unpreconditioned",
           "pc_type": "mg",
           "pc_mg_cycle_type": "v",
           "pc_mg_type": "full",
           "mg_levels_ksp_type": "fgmres",
           "mg_levels_ksp_convergence_test": "skip",
           "mg_levels_ksp_max_it": 6,
           "mg_levels_ksp_norm_type": "unpreconditioned",
           "mg_levels_pc_type": "python",
           "mg_levels_pc_python_type": "firedrake.PatchPC",
           "mg_levels_patch_pc_patch_save_operators": True,
           "mg_levels_patch_pc_patch_partition_of_unity": False,
           "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
           "mg_levels_patch_pc_patch_construct_dim": 0,
           "mg_levels_patch_pc_patch_construct_type": "python",
           "mg_levels_patch_pc_patch_construct_python_type": "alfi.MacroStar",
           "mg_levels_patch_sub_ksp_type": "preonly",
           "mg_levels_patch_sub_pc_type": "lu",
           "mg_levels_patch_sub_pc_factor_mat_solver_type": "umfpack",
           "mg_coarse_ksp_type": "richardson",
           "mg_coarse_ksp_max_it": 1,
           "mg_coarse_ksp_norm_type": "unpreconditioned",
           "mg_coarse_pc_type": "python",
           "mg_coarse_pc_python_type": "firedrake.AssembledPC",
           "mg_coarse_assembled": {
                 "mat_type": "aij",
                 "pc_type": "telescope",
                 "pc_telescope_reduction_factor": 1,
                 "pc_telescope_subcomm_type": "contiguous",
                 "telescope_pc_type": "lu",
                 "telescope_pc_factor_mat_solver_type": "mumps",
                 "telescope_pc_factor_mat_mumps_icntl_14": ICNTL_14,
            }
    },
   }

# LU solver for outer Schur complement
outerschurlu = {
    "ksp_type": "gmres",
    "ksp_max_it": 2,
    "pc_type": "python",
    "pc_python_type": "__main__.SchurPCBE",
    "aux_pc_type": "lu",
    "aux_pc_factor_mat_solver_type": "mumps",
    "aux_mat_mumps_icntl_14": ICNTL_14,
   }

# Main solver
fs3by2 = {
    "snes_type": "newtonls",
    "snes_max_it": 25,
    "snes_linesearch_type": "basic",
    "snes_linesearch_maxstep": 1.0,
    "snes_rtol": 1.0e-10,
    "snes_atol": 1.0e-6,
    "snes_monitor": None,
    "ksp_type": "fgmres",
    "ksp_max_it": 75,
    "ksp_atol": 1.0e-7,
    "ksp_rtol": 1.0e-7,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "mat_type": "aij",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "upper",
    "pc_fieldsplit_0_fields": "0,1,2",
    "pc_fieldsplit_1_fields": "3,4",
}

# Main solver with LU solver for (u,p)-block
fs3by2nslu = {
    "snes_type": "newtonls",
    "snes_max_it": 30,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1.0,
    "snes_rtol": 1.0e-10,
    "snes_atol": 1.0e-7,
    "snes_monitor": None,
    "ksp_type": "fgmres",
    "ksp_max_it": 50,
    "ksp_atol": 1.0e-8,
    "ksp_rtol": 1.0e-9,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "mat_type": "aij",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_0_fields": "0,1,2",
    "pc_fieldsplit_1_fields": "3,4",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_0_mat_mumps_icntl_14": ICNTL_14,
}

# Main solver with LU solver for Schur complement
fs3by2slu = {
    "snes_type": "newtonls",
    "snes_max_it": 30,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1.0,
    "snes_rtol": 1.0e-15,
    "snes_atol": 1.0e-6,
    "snes_monitor": None,
    "ksp_type": "fgmres",
    "ksp_max_it": 40,
    "ksp_atol": 1.0e-7,
    "ksp_rtol": 1.0e-7,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "mat_type": "aij",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_0_fields": "0,1,2",
    "pc_fieldsplit_1_fields": "3,4",
    "fieldsplit_1": outerschurlu,
}

# Main solver with LU solver for (E,B)-block and Schur complement
# Can be used to determine how well the outer Schur complement is approximated by the different linerizations
fs3by2lu = {
    "snes_type": "newtonls",
    "snes_max_it": 30,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1.0,
    "snes_rtol": 1.0e-10,
    "snes_atol": 1.0e-6,
    "snes_monitor": None,
    "ksp_type": "fgmres",
    "ksp_max_it": 50,
    "ksp_atol": 1.0e-7,
    "ksp_rtol": 1.0e-7,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "mat_type": "aij",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_0_fields": "0,1,2",
    "pc_fieldsplit_1_fields": "3,4",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_0_mat_mumps_icntl_14": ICNTL_14,
    "fieldsplit_1": outerschurlu,
}

solvers = {"lu": lu, "fs3by2": fs3by2, "fs3by2nslu": fs3by2nslu, "fs3by2slu": fs3by2slu, "fs3by2lu": fs3by2lu}

# Definition of problem parameters
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--baseN", type=int, default=10)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--Ra", nargs='+', type=float, default=[1])
parser.add_argument("--Pm", nargs='+', type=float, default=[1])
parser.add_argument("--Pr", nargs='+', type=float, default=[1])
parser.add_argument("--gamma", type=float, default=10000)
parser.add_argument("--gamma2", type=float, default=0)
parser.add_argument("--advect", type=float, default=1)
parser.add_argument("--hierarchy", choices=["bary", "uniform"], default="bary")
parser.add_argument("--solver-type", choices=list(solvers.keys()), default="lu")
parser.add_argument("--testproblem", choices=["ldc", "3dgenerator", "smooth"], default="smooth")
parser.add_argument("--discr", choices=["rt", "bdm", "cg"], required=True)
parser.add_argument("--linearisation", choices=["picard", "mdp", "newton"], required=True)
parser.add_argument("--stab", default=False, action="store_true")
parser.add_argument("--checkpoint", default=False, action="store_true")
parser.add_argument("--output", default=False, action="store_true")

args, _ = parser.parse_known_args()
baseN = args.baseN
k = args.k
nref = args.nref
Ra = Constant(args.Ra[0])
Pm = Constant(args.Pm[0])
Pr = Constant(args.Pr[0])
gamma = Constant(args.gamma)
gamma2 = Constant(args.gamma2)
hierarchy = args.hierarchy
solver_type = args.solver_type
testproblem = args.testproblem
gamma2 = Constant(args.gamma2)
advect = Constant(args.advect)
linearisation = args.linearisation
discr = args.discr
stab = args.stab
checkpoint = args.checkpoint
output = args.output

Re = Constant(1)
Rem = Constant(1)
S = Constant(1)


# Stabilisation weight for BurmanStabilisation
stab_weight = Constant(3e-3)

if len(args.Ra) != 1 and len(args.Pm) != 1 and len(args.Pr) != 1:
    raise ValueError("Ra, Pm and Pr cannot all contain more than one element at the same time")

if discr == "cg" and hierarchy != "bary":
    raise ValueError("SV is only stable on barycentric refined grids")

if k < 3 and hierarchy == "bary":
    raise ValueError("Scott Vogelius is not stable for k<3")

# Define the base Mesh for the different problems
if testproblem == "3dgenerator":
    base = BoxMesh(5*baseN, baseN, baseN, 5, 1, 1, distribution_parameters=distribution_parameters)
else:
    base = UnitCubeMesh(baseN, baseN, baseN, distribution_parameters=distribution_parameters)

# Callbacks called before and after mesh refinement
def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)


def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)


# Create Mesh hierarchy
if hierarchy == "bary":
    mh = alfi.BaryMeshHierarchy(base, nref, callbacks=(before, after))
elif hierarchy == "uniformbary":
    bmesh = Mesh(bary(base._plex), distribution_parameters={"partition": False})
    mh = MeshHierarchy(bmesh, nref, reorder=True, callbacks=(before, after),
                       distribution_parameters=distribution_parameters)
elif hierarchy == "uniform":
    mh = MeshHierarchy(base, nref, reorder=True, callbacks=(before, after),
                       distribution_parameters=distribution_parameters)
else:
    raise NotImplementedError("Only know bary, uniformbary and uniform for the hierarchy.")

# Change mesh from [0,1]^3 to [-0.5,0.5]^3
if testproblem != "3dgenerator":
    for m in mh:
        m.coordinates.dat.data[:, 0] -= 0.5
        m.coordinates.dat.data[:, 1] -= 0.5
        m.coordinates.dat.data[:, 2] -= 0.5
mesh = mh[-1]

area = assemble(Constant(1, domain=mh[0])*dx)


def message(msg):
    if mesh.comm.rank == 0:
        warning(msg)


# Define mixed function spaces
if discr == "rt":
    Vel = FiniteElement("N1div", mesh.ufl_cell(), k, variant="integral")
    V = FunctionSpace(mesh, Vel)
elif discr == "bdm":
    Vel = FiniteElement("N2div", mesh.ufl_cell(), k, variant="integral")
    V = FunctionSpace(mesh, Vel)
elif discr == "cg":
    V = VectorFunctionSpace(mesh, "CG", k)
Q = FunctionSpace(mesh, "DG", k-1)  # p
Rel = FiniteElement("N1curl", mesh.ufl_cell(), k, variant="integral")
R = FunctionSpace(mesh, Rel)  # E
Wel = FiniteElement("N1div", mesh.ufl_cell(), k, variant="integral")
W = FunctionSpace(mesh, Wel)
TT = FunctionSpace(mesh, "CG", k)
Z = MixedFunctionSpace([V, Q, TT, W, R])

z = Function(Z)
(u, p, T, B, E) = split(z)
(v, q, s, C, Ff) = split(TestFunction(Z))

# used for BurmanStabilisation
z_last_u = Function(V)

# For continuation we might want to start from checkpoint
if checkpoint:
    try:
        if len(args.Pr) == 1 or len(args.Ra) == 1:
            chk = DumbCheckpoint("dump/"+str(float(S*Pm))+str(linearisation)+str(testproblem), mode=FILE_READ)
        else:
            chk = DumbCheckpoint("dump/"+str(float(Pm))+str(linearisation)+str(testproblem), mode=FILE_READ)
        chk.load(z)
    except Exception as e:
        message(e)
    (u_, p_, B_, E_) = z.split()
    z_last_u.assign(u_)

(x, y, zz) = SpatialCoordinate(Z.mesh())
n = FacetNormal(mesh)

eps = lambda x: sym(grad(x))
g = as_vector([0,0,1])

# Base weak form of problem
F = (
      2/Re * inner(eps(u), eps(v))*dx
    # + advect * inner(dot(grad(u), u), v) * dx
    + gamma * inner(div(u), div(v)) * dx
    + S * inner(cross(B, E), v) * dx
    + S * inner(cross(B, cross(u, B)), v) * dx
    - inner(p, div(v)) * dx
    - inner(div(u), q) * dx
    + inner(E, Ff) * dx
    + inner(cross(u, B), Ff) * dx
    - 1/Pm * inner(B, curl(Ff)) * dx
    + inner(curl(E), C) * dx
    + 1/Pm * inner(div(B), div(C)) * dx
    + gamma2 * inner(div(B), div(C)) * dx
    - Ra/Pr * inner(g*T , v) * dx
    + 1/Pr * inner(grad(T), grad(s)) * dx
    + inner(dot(u, grad(T)), s) * dx
)

# Compute RHS for Method of Manufactured Solution (MMS)


def compute_rhs(u_ex, B_ex, T_ex, p_ex, E_ex):
    E_ex_ = interpolate(E_ex, R)
    f1 = (-2/Re * div(eps(u_ex)) + advect * dot(grad(u_ex), u_ex) - gamma * grad(div(u_ex))
          + grad(p_ex) + S * cross(B_ex, (E_ex + cross(u_ex, B_ex))) - Ra/Pr * g*T_ex)
    f2 = + curl(E_ex_) - 1/Pm * grad(div(B_ex)) - gamma2 * grad(div(B_ex))
    f3 = -1/Pm * curl(B_ex) + E_ex + cross(u_ex, B_ex)
    f4 = -1/Pr * div(grad(T_ex)) + dot(u_ex, grad(T_ex))
    return (f1, f2, f3, f4)

if testproblem == "smooth":
    u_ex = as_vector([cos(zz), sin(x), sin(y)]) 
    B_ex = as_vector([sin(y), cos(zz), sin(x)])
    T_ex = cos(x)
    p_ex = sin(x)
    E_ex = as_vector([sin(x)*cos(x), sin(y), cos(x)]) # 1.0e-13*x
    
    (f1, f2, f3, f4) = compute_rhs(u_ex, B_ex, T_ex, p_ex, E_ex)
    rhs = True  # because RHS is zero

    bcs = [DirichletBC(Z.sub(0), u_ex, "on_boundary"),
           DirichletBC(Z.sub(2), T_ex, "on_boundary"),
           DirichletBC(Z.sub(3), B_ex, "on_boundary"),
           DirichletBC(Z.sub(4), E_ex, "on_boundary"),
           PressureFixBC(Z.sub(1), 0, 1)]

    # Do we know what the exact solution of the problem is?
    solution_known = True

    # Do the boundary conditions depend on parameter thats change during continuation
    bc_varying = False

    # On what ids of the boundary do we want to apply the boundary conditions
    # This is needed for DG-Form of H(div)-L2 formulation
    bcs_ids_apply = (1, 2, 3, 4, 5, 6)
    bcs_ids_dont_apply = None

elif testproblem == "3dgenerator":
    # Hartmann number
    Ha = sqrt(S*Rem*Re)

    # Problem parameter
    B0 = Constant(1.0)
    x_on = Constant(2.0)
    x_off = Constant(2.5)
    delta = Constant(0.1)

    B_z_gen = (B0/2)*(tanh((x-x_on)/delta) - tanh((x-x_off)/delta))

    # *_ex are just defined for boundary conditions. We don't know exact solution
    u_ex = Constant((1, 0, 0), domain=mesh)
    B_ex = as_vector([Constant(0, domain=mesh), Constant(0, domain=mesh), B_z_gen])
    E_ex = 1/Rem * curl(B_ex)
    p_ex = Constant(0, domain=mesh)
    T_ex = sin(x)

    # Inflow BCs for u at x=0, outflow at x=1, no-slip on 4 sides

    # On what ids of the boundary do we want to apply the boundary conditions
    # This is needed for DG-Form of H(div)-L2 formulation
    bcs_ids_apply = (1)
    bcs_ids_dont_apply = (3, 4, 5, 6)

    bcs = [DirichletBC(Z.sub(0), u_ex, bcs_ids_apply),
           DirichletBC(Z.sub(0), Constant((0., 0., 0.)), bcs_ids_dont_apply),
           DirichletBC(Z.sub(2), T_ex, "on_boundary"),
           DirichletBC(Z.sub(3), B_ex, "on_boundary"),
           DirichletBC(Z.sub(4), 0, "on_boundary"),
           PressureFixBC(Z.sub(1), 0, 1)]

    rhs = None  # because rhs is zero for this problem

    # Do we know what the exact solution of the problem is?
    solution_known = False

    # Do the boundary conditions depend on parameter thats change during continuation
    bc_varying = True


elif testproblem == "ldc":
    # example taken from https://doi.org/10.1016/j.jcp.2016.04.019
    u_ex = Constant((0, 0, 0), domain=mesh)
    B_ex = Constant((0, 1, 0), domain=mesh)
    B_ex = project(B_ex, W)

    bcs_ids_apply = (1, 2, 3, 4, 5, 6)
    bcs_ids_dont_apply = None

    bcs = [DirichletBC(Z.sub(0), u_ex, bcs_ids_apply),  # 4 == upper boundary (y==1)
           DirichletBC(Z.sub(2), 1, 2),
           DirichletBC(Z.sub(2), 0, 1),
           DirichletBC(Z.sub(3), B_ex, "on_boundary"),
           DirichletBC(Z.sub(4), 0, "on_boundary"),
           PressureFixBC(Z.sub(1), 0, 1)]
    rhs = None
    solution_known = False
    bc_varying = False

if solution_known:
    u_ex_ = interpolate(u_ex, V)
    B_ex_ = interpolate(B_ex, W)
    T_ex_ = interpolate(T_ex, TT)
    p_ex_ = interpolate(p_ex, Q)
    E_ex_ = interpolate(E_ex, R)

# Add Burman Stabilisation
if stab:
    initial = interpolate(as_vector([sin(y), x, x]), V)
    z_last_u.assign(initial)
    stabilisation = BurmanStabilisation(Z.sub(0), state=z_last_u, h=FacetArea(mesh), weight=stab_weight)
    stabilisation_form_u = stabilisation.form(u, v)
    F += (advect * stabilisation_form_u)

# For H(div)-L2 discretization we have to add a DG-formulation of the advection and diffusion term
if discr in ["rt", "bdm"]:
    h = CellVolume(mesh)/FacetArea(mesh)
    sigma = Constant(1) * Z.sub(0).ufl_element().degree()**2  # penalty term
    theta = Constant(1)  # theta=1 means upwind discretization of nonlinear advection term
    uflux_int = 0.5*(dot(u, n) + theta*abs(dot(u, n)))*u
    uflux_ext_1 = 0.5*(inner(u, n) + theta*abs(inner(u, n)))*u
    uflux_ext_2 = 0.5*(inner(u, n) - theta*abs(inner(u, n)))*u_ex

    F_DG = (
         - 1/Re * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS
         - 1/Re * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS
         + 1/Re * sigma/avg(h) * inner(2*avg(outer(u, n)), 2*avg(outer(v, n))) * dS
         - inner(outer(v, n), 2/Re*sym(grad(u))) * ds
         - inner(outer(u-u_ex, n), 2/Re*sym(grad(v))) * ds(bcs_ids_apply)
         + 1/Re*(sigma/h)*inner(v, u-u_ex) * ds(bcs_ids_apply)
         - advect * dot(u, div(outer(v, u))) * dx
         + advect * dot(v('+')-v('-'), uflux_int('+')-uflux_int('-')) * dS
         + advect * dot(v, uflux_ext_1) * ds
         + advect * dot(v, uflux_ext_2) * ds(bcs_ids_apply)
    )

    if bcs_ids_dont_apply is not None:
        F_DG += (
            - inner(outer(u, n), 2/Re*sym(grad(v))) * ds(bcs_ids_dont_apply)
            + 1/Re*(sigma/h)*inner(v, u) * ds(bcs_ids_dont_apply)
           )

    F += F_DG

elif discr == "cg":
    F += advect * inner(dot(grad(u), u), v) * dx

if rhs is not None:
    F -= inner(f1, v) * dx + inner(f3, Ff) * dx + inner(f2, C) * dx + inner(f4, s) * dx

# Definition of the three different linearizations
w = TrialFunction(Z)
[w_u, w_p, w_T, w_B, w_E] = split(w)

J_newton = ufl.algorithms.expand_derivatives(derivative(F, z, w))

if linearisation == "newton":
    J = J_newton

elif linearisation == "mdp":
    J_mdp = (
          J_newton
        - inner(cross(w_u, B), Ff) * dx  # G
            )
    J = J_mdp

elif linearisation == "picard":
    J_picard = (
          J_newton
        - S * inner(cross(w_B, E), v) * dx  # J_tilde
        - S * inner(cross(w_B, cross(u, B)), v) * dx  # D_1_tilde
        - S * inner(cross(B, cross(u, w_B)), v) * dx  # D_2_tilde
        - inner(cross(u, w_B), Ff) * dx  # G_tilde
            )
    J = J_picard

else:
    raise ValueError("only know newton, mdp and picard as linearisation method")

problem = NonlinearVariationalProblem(F, z, bcs, J=J)

appctx = {"Ra": Ra, "gamma": gamma, "nu": 1/Ra, "Pm": Pm, "gamma2": gamma2}
params = solvers[args.solver_type]

# Depending on the Mesh Hierarchy we have to use star or macrostar solver
if args.solver_type in ["fs3by2", "fs3by2slu"]:
    params["fieldsplit_0"] = nsfsstar if hierarchy == "uniform" else nsfsmacrostar

if args.solver_type in ["fs3by2", "fs3by2mlu"]:
    params["fieldsplit_1"] = outerschurstar if hierarchy == "uniform" else outerschurmacrostar

if args.hierarchy == "bary":
    params["snes_linesearch_type"] = "l2"

# Definition of solver and transfer operators
solver = NonlinearVariationalSolver(problem, solver_parameters=params, options_prefix="", appctx=appctx)
qtransfer = NullTransfer()
Etransfer = NullTransfer()
vtransfer = SVSchoeberlTransfer((1/Re, gamma), 2, hierarchy)
dgtransfer = DGInjection()

transfers = {
             Q.ufl_element(): (prolong, restrict, qtransfer.inject),
             VectorElement("DG", mesh.ufl_cell(), args.k): (dgtransfer.prolong, restrict, dgtransfer.inject),
             VectorElement("DG", mesh.ufl_cell(), args.k-1): (dgtransfer.prolong, restrict, dgtransfer.inject),
            }

# On barycentric refined grids we need special prolongation operators
if hierarchy == "bary":
    transfers[V.ufl_element()] = (vtransfer.prolong, vtransfer.restrict, inject)

transfermanager = TransferManager(native_transfers=transfers)
solver.set_transfer_manager(transfermanager)

results = {}
Ras = args.Ra
Pms = args.Pm
Prs = args.Pr

pvd = File("output/mhd.pvd")

def run(ra, pm, pr):
    (u, p, T, B, E) = z.split()
    Ra.assign(ra)
    Pm.assign(pm)
    Pr.assign(pr)

    # Indices for output depending on Re-S, Re-Rem or S-Rem table
    if len(args.Pr) == 1 or len(args.Pm) == 1:
        ind1 = ra
        ind2 = pr*pm
    else:
        ind1 = ra*pr
        ind2 = pm

    if bc_varying:
        global p_ex
        u_ex_ = interpolate(u_ex, V)
        B_ex_ = interpolate(B_ex, W)
        T_ex_ = interpolate(T_ex, TT)
        E_ex_ = interpolate(E_ex, R)
        pintegral = assemble(p_ex*dx)
        p_ex = p_ex - Constant(pintegral/area)
        # bcs[0] is u, bcs[1] is B, bcs[2] is E
        bcs[0].function_arg = u_ex_

    if checkpoint:
        try:
            chk = DumbCheckpoint("dump/"+str(float(ind2))+str(linearisation)+str(testproblem), mode=FILE_READ)
            chk.load(z)
        except Exception as e:
            message(e)

    if mesh.comm.rank == 0:
        print(GREEN % ("Solving for #dofs = %s, Ra = %s, Pm = %s, gamma = %s, Pr = %s, baseN = %s, nref = %s, "
                       "linearisation = %s, testproblem = %s, discr = %s, k = %s"
                       % (Z.dim(), float(ra), float(pm), float(gamma), float(pr), int(baseN), int(nref),
                          linearisation, testproblem, discr, int(float(k)))), flush=True)
    if not os.path.exists("results/"):
        os.mkdir("results/")

    # Update z_last_u in Burman Stabilisation
    if stab:
        stabilisation.update(z.split()[0])
        z_last_u.assign(u)

    # For the continuation in Re we set up the solver again
    if float(Re) == 100:
        if mesh.comm.rank == 0:
            print("Setting up new solver", flush=True)
        global solver
        problem = NonlinearVariationalProblem(F, z, bcs, J=J)
        solver = NonlinearVariationalSolver(problem, solver_parameters=params, options_prefix="", appctx=appctx)
        solver.set_transfer_manager(transfermanager)

    # Solve the problem and measure time
    start = datetime.now()
    solver.solve()
    end = datetime.now()

    # Iteration numbers
    linear_its = solver.snes.getLinearSolveIterations()
    nonlinear_its = solver.snes.getIterationNumber()
    time = (end-start).total_seconds() / 60

    if mesh.comm.rank == 0:
        if nonlinear_its == 0:
            nonlinear_its = 1
        print(GREEN % ("Time taken: %.2f min in %d nonlinear iterations, %d linear iterations (%.2f Krylov iters per Newton step)"
                       % (time, nonlinear_its, linear_its, linear_its/float(nonlinear_its))), flush=True)
        print("%.2f @ %d @ %d @ %.2f" % (time, nonlinear_its, linear_its, linear_its/float(nonlinear_its)), flush=True)

    (u, p, T, B, E) = z.split()

    # Make sure that average of p is 0
    pintegral = assemble(p*dx)
    p.assign(p - Constant(pintegral/area))
    B.rename("MagneticField")
    u.rename("VelocityField")
    T.rename("Temperature")
    p.rename("Pressure")
    E.rename("ElectricFieldf")

    # Compute divergence of u and B
    norm_div_u = sqrt(assemble(inner(div(u), div(u))*dx))
    norm_div_B = sqrt(assemble(inner(div(B), div(B))*dx))

    if mesh.comm.rank == 0:
        print("||div(u)||_L^2 = %s" % norm_div_u, flush=True)
        print("||div(B)||_L^2 = %s" % norm_div_B, flush=True)

    if solution_known:
        B_ex_ = interpolate(B_ex, B.function_space())
        u_ex_ = interpolate(u_ex, u.function_space())
        T_ex_ = interpolate(T_ex, T.function_space())
        p_ex_ = interpolate(p_ex, p.function_space())
        E_ex_ = interpolate(E_ex, E.function_space())
        B_ex_.rename("ExactSolutionB")
        u_ex_.rename("ExactSolutionu")
        T_ex_.rename("ExactSolutionT")
        p_ex_.rename("ExactSolutionp")
        E_ex_.rename("ExactSolutionE")

        # Compute error for MMS
        error_u = errornorm(u_ex, u, 'L2')
        error_B = errornorm(B_ex, B, 'L2')
        error_T = errornorm(T_ex, T, 'L2')
        error_E = errornorm(E_ex, E, 'L2')
        error_p = errornorm(p_ex, p, 'L2')

        if mesh.comm.rank == 0:
            print("Error ||u_ex - u||_L^2 = %s" % error_u, flush=True)
            print("Error ||p_ex - p||_L^2 = %s" % error_p, flush=True)
            print("Error ||T_ex - T||_L^2 = %s" % error_T, flush=True)
            print("Error ||B_ex - B||_L^2 = %s" % error_B, flush=True)
            print("Error ||E_ex - E||_L^2 = %s" % error_E, flush=True)

            # Write errors to file
            f = open("error.txt", 'a+')
            f.write("%s,%s,%s,%s,%s\n" % (error_u, error_p, error_T, error_B, error_E))
            f.close()

        # Save plots of solution
        if output:
            File("output/mhd.pvd").write(u, u_ex_, p, p_ex_, T, T_ex_, B, B_ex_, E, E_ex_)

        sys.stdout.flush()
        info_dict = {
            "Ra": ra,
            "Pm": pm,
            "Pr": pr,
            "krylov/nonlin": linear_its/nonlinear_its,
            "nonlinear_iter": nonlinear_its,
            "error_u": error_u,
            "error_p": error_p,
            "error_T": error_T,
            "error_B": error_B,
            "error_E": error_E,
        }

    else:
        info_dict = {
            "Ra": ra,
            "Pm": pm,
            "Pr": pr,
            "krylov/nonlin": linear_its/nonlinear_its,
            "nonlinear_iter": nonlinear_its,
        }

        if output:
            pvd.write(u, p, T, B, E, time=float(ind1)*float(ind2))

    message(BLUE % info_dict)
    # Write iteration numbers to file
    if mesh.comm.rank == 0:
        dir = 'results/results'+str(linearisation)+str(testproblem)+'/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        f = open(dir+str(float(ind1))+str(float(ind2))+'.txt', 'w+')
        f.write("({0:2.0f}){1:4.1f}".format(float(info_dict["nonlinear_iter"]), float(info_dict["krylov/nonlin"])))
        f.close()

    z_last_u.assign(u)

    # Create Checkpoints of solution
    if not os.path.exists("dump/"):
        os.mkdir("dump/")
    chk = DumbCheckpoint("dump/"+str(float(ind2))+str(linearisation)+str(testproblem), mode=FILE_CREATE)
    chk.store(z)


# Loop over parameters
for pm in Pms:
    for pr in Prs:
        for ra in Ras:
            try:
                run(ra, pm, pr)
            # If solve fails report 0 as iteration number
            except Exception as e:
                message(e)
                dir = 'results/results'+str(linearisation)+str(testproblem)+'/'
                if not os.path.exists(dir):
                    os.mkdir(dir)
                if len(args.Pr) == 1 or len(args.Pm) == 1:
                    f = open(dir+str(float(ra))+str(float(pm*pr))+'.txt', 'w+')
                else:
                    f = open(dir+str(float(ra*pr))+str(float(pm))+'.txt', 'w+')
                f.write("({0:2.0f}){1:4.1f}".format(0, 0))
                f.close()
