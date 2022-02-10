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
from mpi4py import MPI
import gc
gc.disable()

import petsc4py
petsc4py.PETSc.Sys.popErrorHandler()

# Parallel distribution parameters
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

# Define two-dimensional versions of cross and curl operators
def scross(x, y):
    return x[0]*y[1] - x[1]*y[0]

def vcross(x, y):
    return as_vector([x[1]*y, -x[0]*y])

def scurl(x):
    return x[1].dx(0) - x[0].dx(1)

def vcurl(x):
    return as_vector([x.dx(1), -x.dx(0)])

def acurl(x):
    return as_vector([
                     x[2].dx(1),
                     -x[2].dx(0),
                     x[1].dx(0) - x[0].dx(1)
                     ])

# Definition of Burman Stabilisation term to avoid oscillations in velocity u
def BurmanStab(B, C, wind, stab_weight, mesh):
    n = FacetNormal(mesh)
    h = FacetArea(mesh)
    beta = avg(facet_avg(sqrt(inner(wind, wind)+1e-10)))
    gamma1 = stab_weight  # as chosen in doi:10.1016/j.apnum.2007.11.001
    stabilisation_form = 0.5 * gamma1 * avg(h)**2 * beta * dot(jump(grad(B), n), jump(grad(C), n))*dS
    return stabilisation_form


# Definition of outer Schur complement for the order ((E,B),(u,p)), which is Navier-Stokes block + Lorentz force
class SchurPCup(AuxiliaryOperatorPC):
    def form(self, pc, V, U):
        mesh = V.function_space().mesh()
        [u, p] = split(U)
        [v, q] = split(V)
        Z = V.function_space()
        U_ = Function(Z)

        state = self.get_appctx(pc)['state']
        [u_n, p_n, B_n, E_n] = split(state)

        eps = lambda x: sym(grad(x))

        # Factor for Schur complement approximation as chosen in doi.org/10.1137/16M1074084
        if linearisation == "mdp":
            alpha = 1
        elif linearisation == "picard":
            alpha = dt/(dt+Rem*(1/(baseN*nref))**2)
        elif linearisation == "newton":
            norm_un = sqrt(assemble(inner(u_n, u_n)*dx))
            hh = 1/(baseN*nref)
            alpha = dt/(dt + Rem*hh**2 + Rem*hh*norm_un*dt)

        # Weak form of NS + (scaled) Lorentz force
        A = (
              + 2/Re * inner(eps(u), eps(v))*dx
              + gamma * inner(div(u), div(v)) * dx
              - inner(p, div(v)) * dx
              - inner(div(u), q) * dx
              + alpha*S * inner(vcross(B_n, scross(u, B_n)), v) * dx
        )

        # For H(div)-L2 discretization we have to add a DG-formulation of the advection and diffusion term
        if discr in ["rt", "bdm"]:
            h = CellVolume(mesh)/FacetArea(mesh)
            uflux_int = 0.5*(dot(u, n) + abs(dot(u, n)))*u
            uflux_ext_1 = 0.5*(inner(u, n) + abs(inner(u, n)))*u
            uflux_ext_2 = 0.5*(inner(u, n) - abs(inner(u, n)))*u_ex

            A_DG = (
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

        A = inner(u, v) * dx + dt_factor*dt*A
        return (A, bcs)


# Definition of outer Schur complement for the order ((u,p),(B,E))
class SchurPCBE(AuxiliaryOperatorPC):
    def form(self, pc, V, U):
        [B, E] = split(U)
        [C, Ff] = split(V)
        state = self.get_appctx(pc)['state']
        [u_n, p_n, B_n, E_n] = split(state)

        A = (
             + 1*inner(E, Ff) * dx
             + inner(scross(u_n, B), Ff) * dx
             - 1/Rem * inner(B, vcurl(Ff)) * dx
             + inner(vcurl(E), C) * dx
             + 1/Rem * inner(div(B), div(C)) * dx
             + gamma2 * inner(div(B), div(C)) * dx
                      )

        bcs = [DirichletBC(V.function_space().sub(0), 0, "on_boundary"),
               DirichletBC(V.function_space().sub(1), 0, "on_boundary"),
               ]

        A = inner(B, C) * dx + dt_factor*dt*A
        return (A, bcs)


class PressureFixBC(DirichletBC):
    def __init__(self, V, val, subdomain, method="topological"):
        super().__init__(V, val, subdomain, method)
        sec = V.dm.getDefaultSection()
        dm = V.mesh().topology_dm

        coordsSection = dm.getCoordinateSection()
        coordsDM = dm.getCoordinateDM()
        dim = coordsDM.getDimension()
        coordsVec = dm.getCoordinatesLocal()

        (vStart, vEnd) = dm.getDepthStratum(0)
        indices = []
        for pt in range(vStart, vEnd):
            x = dm.getVecClosure(coordsSection, coordsVec, pt).reshape(-1, 3).mean(axis=0)
            if (x[0] == 1.0) and (x[1] == 0.0) and (x[2] == 0.0): # fix [1,0, 0] in unmapped mesh coordinates (bottom left corner)

                if dm.getLabelValue("pyop2_ghost", pt) == -1:
                    indices = [pt]
                break

        # Check if this worked!
        if V.mesh().comm.size == 1 and len(indices) != 1: assert False

        nodes = []
        for i in indices:
            if sec.getDof(i) > 0:
                nodes.append(sec.getOffset(i))

        self.nodes = numpy.asarray(nodes, dtype=IntType)
        if len(self.nodes) > 0:
            sys.stdout.write("Fixing nodes %s on rank %d on dim %d\n" % (self.nodes, mesh.comm.rank, V.dim()))
            sys.stdout.flush()


# Definition of different solver parameters

# Increase buffer size for MUMPS solver
ICNTL_14 = 5000
tele_reduc_fac = int(MPI.COMM_WORLD.size/4)
if tele_reduc_fac < 1:
    tele_reduc_fac = 1

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
#         "mg_levels_pc_star_backend": "tinyasm",
         "mg_levels_pc_star_construct_dim": 0,
         "mg_levels_pc_star_sub_sub_ksp_type": "preonly",
         "mg_levels_pc_star_sub_sub_pc_type": "lu",
         "mg_levels_pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack",
         "mg_coarse_ksp_type": "richardson",
         "mg_coarse_ksp_max_it": 1,
         "mg_coarse_ksp_norm_type": "unpreconditioned",
         "mg_coarse_pc_type": "python",
         "mg_coarse_pc_python_type": "firedrake.AssembledPC",
         "mg_coarse_assembled": {
             "mat_type": "aij",
             "pc_type": "telescope",
             "pc_telescope_reduction_factor": tele_reduc_fac,
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
                 "pc_telescope_reduction_factor": tele_reduc_fac,
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
#         "mg_levels_pc_python_type": "firedrake.ASMStarPC",
#         "mg_levels_pc_star_backend": "tinyasm",
         "mg_levels_pc_python_type": "firedrake.PatchPC",
         "mg_levels_patch_pc_patch_save_operators": True,
         "mg_levels_patch_pc_patch_partition_of_unity": False,
         "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
         "mg_levels_patch_pc_patch_construct_dim": 0,
         "mg_levels_patch_pc_patch_construct_type": "star",
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
               "pc_telescope_reduction_factor": tele_reduc_fac,
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
                 "pc_telescope_reduction_factor": tele_reduc_fac,
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
fs2by2 = {
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
    "pc_fieldsplit_0_fields": "0,1",
    "pc_fieldsplit_1_fields": "2,3",
}

# Main solver with LU solver for (u,p)-block
fs2by2nslu = {
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
    "pc_fieldsplit_0_fields": "0,1",
    "pc_fieldsplit_1_fields": "2,3",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_0_mat_mumps_icntl_14": ICNTL_14,
}

# Main solver with LU solver for Schur complement
fs2by2slu = {
    "snes_type": "newtonls",
    "snes_max_it": 30,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1.0,
    "snes_rtol": 1.0e-15,
    "snes_atol": 1.0e-7,
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
    "pc_fieldsplit_0_fields": "0,1",
    "pc_fieldsplit_1_fields": "2,3",
    "fieldsplit_1": outerschurlu,
}

# Main solver with LU solver for (E,B)-block and Schur complement
# Can be used to determine how well the outer Schur complement is approximated by the different linerizations
fs2by2lu = {
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
    "pc_fieldsplit_0_fields": "0,1",
    "pc_fieldsplit_1_fields": "2,3",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_0_mat_mumps_icntl_14": ICNTL_14,
    "fieldsplit_1": outerschurlu,
}

solvers = {"lu": lu, "fs2by2": fs2by2, "fs2by2nslu": fs2by2nslu, "fs2by2slu": fs2by2slu, "fs2by2lu": fs2by2lu}

# Making a mesh hierarchy for multigrid in the periodic case is currently a little
# complicated in firedrake.
# Background: firedrake makes an x-periodic mesh by first meshing the surface of
# a cylinder, then appling a coordinate transformation.
# To make a mesh hierarchy of periodic meshes, we have to make our own hierarchy of
# meshes of the cylinder, then periodise each one.
QuadMesh = False
def periodise(m):
    coord_fs = VectorFunctionSpace(m, "DG" if not QuadMesh else "DQ", 1, dim=2)
    old_coordinates = m.coordinates
    new_coordinates = Function(coord_fs)

    # make x-periodic mesh
    # unravel x coordinates like in periodic interval
    # set y coordinates to z coordinates
    domain = "{[i, j]: 0 <= i < old_coords.dofs and 0 <= j < new_coords.dofs}"
    instructions = """
    <float64> Y = 0
    <float64> pi = 3.141592653589793
    for i
        Y = Y + old_coords[i, 1]
    end
    for j
        new_coords[j, 0] = atan2(old_coords[j, 1], old_coords[j, 0]) / (pi* 2)
        new_coords[j, 0] = if(new_coords[j, 0] < 0, new_coords[j, 0] + 1, new_coords[j, 0])
        new_coords[j, 0] = if(new_coords[j, 0] == 0 and Y < 0, 1, new_coords[j, 0])
        new_coords[j, 0] = new_coords[j, 0] * Lx[0]
        new_coords[j, 1] = old_coords[j, 2] * Ly[0]
    end
    """

    cLx = Constant(1)
    cLy = Constant(1)

    par_loop((domain, instructions), dx,
             {"new_coords": (new_coordinates, WRITE),
              "old_coords": (old_coordinates, READ),
              "Lx": (cLx, READ),
              "Ly": (cLy, READ)},
             is_loopy_kernel=True)

    return Mesh(new_coordinates)


# Fix errors in mesh coordinates; we know the coordinates
# of the vertices should live on a certain equispaced grid,
# so snap to that
def snap(mesh, N, L=1):
    coords = mesh.coordinates.dat.data
    coords[...] = numpy.round((N/L)*coords)*(L/N)


# Define mesh
def make_mesh(baseN, nref):
    if QuadMesh:
        diagonal = None
        snapfactor = 1
    else:
        diagonal = "crossed"
        snapfactor = 2

    # Need a hierarchy of periodic meshes, so create a hierarchy of cylinders, then unfold
    base = CylinderMesh(baseN, baseN, 1.0, 1.0,
                        longitudinal_direction="z", quadrilateral=QuadMesh,
                        diagonal=diagonal,
                        distribution_parameters=distribution_parameters)

    # Callbacks called before and after mesh refinement.
    # This is where we rebalance to try to equidistribute vertices,
    # for better parallel load balancing.
    def before(dm, i): pass
    def after(dm, i):
        try:
            dm.rebalanceSharedPoints(useInitialGuess=False, parallel=False, entityDepth=0)
        except RuntimeError:
            warning("Vertex rebalancing from scratch failed on level %i" % i)
        try:
            dm.rebalanceSharedPoints(useInitialGuess=True, parallel=True, entityDepth=0)
        except RuntimeError:
            warning("Vertex rebalancing from initial guess failed on level %i" % i)

    # refine nref times
    mh = MeshHierarchy(base, nref,
                       distribution_parameters=distribution_parameters,
                       callbacks=(before, after)
                      )

    # Run everybody through Periodise
    meshes = tuple(periodise(m) for m in mh)
    # Reconstruct hierarchy
    mh = HierarchyBase(meshes, mh.coarse_to_fine_cells, mh.fine_to_coarse_cells, nested=True)

    # Snap coordinates
    for (i, m) in enumerate(mh):
        if i > 0:
            snap(m, baseN * 2**i * snapfactor)

    # Map onto [-1,1]^2
    for mesh in mh:
        mesh.coordinates.dat.data[...] = 2*mesh.coordinates.dat.data - 1

    return mh[-1]

# Definition of problem paramters
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--baseN", type=int, default=10)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--Re", nargs='+', type=float, default=[1])
parser.add_argument("--Rem", nargs='+', type=float, default=[1])
parser.add_argument("--gamma", type=float, default=10000)
parser.add_argument("--gamma2", type=float, default=0)
parser.add_argument("--advect", type=float, default=1)
parser.add_argument("--dt", type=float, required=True)
parser.add_argument("--Tf", type=float, default=1)
parser.add_argument("--S", nargs='+', type=float, default=[1])
parser.add_argument("--hierarchy", choices=["bary", "uniform"], default="bary")
parser.add_argument("--discr", choices=["rt", "bdm", "cg"], required=True)
parser.add_argument("--solver-type", choices=list(solvers.keys()), default="lu")
parser.add_argument("--testproblem", choices=["ldc", "hartmann", "island_coal", "smooth"], default="Wathen")
parser.add_argument("--linearisation", choices=["picard", "mdp", "newton"], required=True)
parser.add_argument("--stab", default=False, action="store_true")
parser.add_argument("--output", default=False, action="store_true")

args, _ = parser.parse_known_args()
baseN = args.baseN
k = args.k
nref = args.nref
Re = Constant(args.Re[0])
Rem = Constant(args.Rem[0])
gamma = Constant(args.gamma)
gamma2 = Constant(args.gamma2)
S = Constant(args.S[0])
hierarchy = args.hierarchy
discr = args.discr
solver_type = args.solver_type
testproblem = args.testproblem
gamma2 = Constant(args.gamma2)
advect = Constant(args.advect)
dt = Constant(args.dt)
Tf = Constant(args.Tf)
linearisation = args.linearisation
stab = args.stab
output = args.output

# Stabilisation weight for BurmanStabilisation
stab_weight = Constant(3e-3)

myk = Constant(0.2)
pert = Constant(-0.01)

if len(args.Re) != 1 and len(args.Rem) != 1 and len(args.S) != 1:
    raise ValueError("Re, Rem and S cannot all contain more than one element at the same time")

if discr == "cg" and hierarchy != "bary":
    raise ValueError("SV is only stable on barycentric refined grids")

if float(Rem) != 1 and testproblem == "hartmann":
    raise ValueError("Don't use hartmann for Rem!=1, because then the rhs is not 0")


if testproblem != "island_coal":
    base = UnitSquareMesh(baseN, baseN, diagonal="crossed", distribution_parameters=distribution_parameters)

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
    for m in mh:
        m.coordinates.dat.data[:, 0] -= 0.5
        m.coordinates.dat.data[:, 1] -= 0.5
    mesh = mh[-1]
else:
    mesh = make_mesh(baseN, nref)
    
area = assemble(Constant(1, domain=mesh)*dx)

def message(msg):
    if mesh.comm.rank == 0:
        warning(msg)


# Define mixed function space
if discr == "rt":
    Vel = FiniteElement("N1div", mesh.ufl_cell(), k, variant=f"integral({k+1})")
    V = FunctionSpace(mesh, Vel)
elif discr == "bdm":
    Vel = FiniteElement("N2div", mesh.ufl_cell(), k, variant=f"integral({k+1})")
    V = FunctionSpace(mesh, Vel)
elif discr == "cg":
    V = VectorFunctionSpace(mesh, "CG", k)
Q = FunctionSpace(mesh, "DG", k-1)  # p
R = FunctionSpace(mesh, "CG", k)  # E
Wel = FiniteElement("N1div", mesh.ufl_cell(), k, variant=f"integral({k+1})")
W = FunctionSpace(mesh, Wel)
Z = MixedFunctionSpace([V, Q, W, R])

# used for BurmanStabilisation
z_last_u = Function(V)
# time variable
t = Constant(0)

(x, y) = SpatialCoordinate(Z.mesh())
n = FacetNormal(mesh)
tangential = as_vector([n[1], -n[0]])

if testproblem == "ldc":
    # example taken from https://doi.org/10.1016/j.jcp.2016.04.019
    u_ex = Constant((1, 0), domain=mesh)
    B_ex = Constant((0, 1), domain=mesh)
    B_ex = interpolate(B_ex, W)
    E_ex = Constant(0, domain=mesh)
    E_ex = interpolate(E_ex, R)
    p_ex = Constant(0, domain=mesh)

    # On what ids of the boundary do we want to apply the boundary conditions
    # This is needed for DG-Form of H(div)-L2 formulation
    bcs_ids_apply = 4
    bcs_ids_dont_apply = (1, 2, 3)

    bcs = [DirichletBC(Z.sub(0), u_ex, bcs_ids_apply),  # 4 == upper boundary (y==1)
           DirichletBC(Z.sub(0), 0, bcs_ids_dont_apply),
           DirichletBC(Z.sub(2), B_ex, "on_boundary"),
           DirichletBC(Z.sub(3), 0, "on_boundary"),
           PressureFixBC(Z.sub(1), 0, 1)]

    rhs = None

    # Do we know what the exact solution of the problem is?
    solution_known = False

    # Do the boundary conditions depend on parameters
    bc_varying = False

elif testproblem == "island_coal":
    b = (cosh(2*pi*y) + myk*cos(2*pi*x))
    u_ex = Constant([0, 0], domain=mesh)
    p_ex = ((1-myk*myk)/2)*(1 + 1/(b*b))
    B_ex = as_vector([sinh(2*pi*y)/b, myk*sin(2*pi*x)/b])
#    E_ex = Constant(0, domain=mesh)
    E_ex = 1/Rem * scurl(B_ex) - scross(u_ex, B_ex)

    B_pert = (pert/pi)*as_vector([-cos(pi*x)*sin(pi*y/2),2*cos(pi*y/2)*sin(pi*x)])

    bcs_ids_apply = "on_boundary"
    bcs_ids_dont_apply = None

    bcs = [DirichletBC(Z.sub(0), u_ex, bcs_ids_apply),  # 4 == upper boundary (y==1)
           DirichletBC(Z.sub(2), B_ex, "on_boundary"),
           DirichletBC(Z.sub(3), E_ex, "on_boundary"),
           PressureFixBC(Z.sub(1), 0, 1)]

    rhs = True
    denom = cosh(2*pi*y) + myk*cos(2*pi*x)
    coef = -(8*pi*pi*(myk*myk-1))/(Rem*denom*denom*denom)
    B_rhs = as_vector([coef*sinh(2*pi*y),coef*myk*sin(2*pi*x)])
    B_rhs = interpolate(B_rhs, Z.sub(2))
    
    # Do we know what the exact solution of the problem is?
    solution_known = False

    # Do the boundary conditions depend on parameters
    bc_varying = False

elif testproblem == "smooth":
    u_ex = as_vector([sin(y), cos(x)])
    p_ex = sin(x)
    B_ex = Constant((1,0), domain=mesh)#as_vector([sinh(2*pi*y), sin(2*pi*x)])
    #    E_ex = Constant(0, domain=mesh)
    E_ex = Constant(0, domain=mesh)

    bcs_ids_apply = (1,2,3,4)
    bcs_ids_dont_apply = None

    bcs = [DirichletBC(Z.sub(0), u_ex, bcs_ids_apply),  # 4 == upper boundary (y==1)
           DirichletBC(Z.sub(2), B_ex, "on_boundary"),
           DirichletBC(Z.sub(3), E_ex, "on_boundary"),
           PressureFixBC(Z.sub(1), 0, 1)]


    B_rhs = interpolate(as_vector([exp(y), sin(x)]), Z.sub(2))
    rhs = True

    # Do we know what the exact solution of the problem is?
    solution_known = False

    # Do the boundary conditions depend on parameters
    bc_varying = False


# Cell diameter used in DG formulation
h = CellVolume(mesh)/FacetArea(mesh)
# Penalty parameter used in DG formulation
sigma = Constant(10) * Z.sub(0).ufl_element().degree()**2

theta = Constant(1)

# Definition of weak form
# timelevel = 0 is time we solve for; timelevel = 1 means previous timestep
def form(z, test_z, Z, timelevel):
    (u, p, B, E) = split(z)
    (v, q, C, Ff) = split(test_z)
    eps = lambda x: sym(grad(x))
    F = (
        + 2/Re * inner(eps(u), eps(v))*dx
        # + advect * inner(dot(grad(u), u), v) * dx
        + gamma * inner(div(u), div(v)) * dx
        + S * inner(vcross(B, E), v) * dx
        + S * inner(vcross(B, scross(u, B)), v) * dx
        - inner(p, div(v)) * dx
        - inner(div(u), q) * dx
        + inner(E, Ff) * dx
        + inner(scross(u, B), Ff) * dx
        - 1/Rem * inner(B, vcurl(Ff)) * dx
        + inner(vcurl(E), C) * dx
        + 1/Rem * inner(div(B), div(C)) * dx
        + gamma2 * inner(div(B), div(C)) * dx
    )

    if discr in ["rt", "bdm"]:
        uflux_int = 0.5*(dot(u, n) + theta*abs(dot(u, n))) * u
        uflux_ext_1 = 0.5*(inner(u, n) + theta*abs(inner(u, n))) * u
        uflux_ext_2 = 0.5*(inner(u, n) - theta*abs(inner(u, n))) * u_ex

        F_DG = (
             - 1/Re * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS
             - 1/Re * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS
             + 1/Re * sigma/avg(h) * inner(2*avg(outer(u, n)), 2*avg(outer(v, n))) * dS
             - inner(outer(v, n), 2/Re*sym(grad(u))) * ds
             - advect * dot(u, div(outer(v, u))) * dx
             + advect * dot(v('+')-v('-'), uflux_int('+')-uflux_int('-')) * dS
             + advect * dot(v, uflux_ext_1) * ds
        )

        # Only add boundary conditions on timelevel we solve for
        # Otherwise we run into problems if initial guess doesn't fullfill right boundary conditions
        if testproblem == "island_coal":
            my_ds = ds
        else:
            my_ds = ds(bcs_ids_apply) 
        
        if timelevel == 0:
            F_DG += (
                 - inner(outer(u-u_ex, n), 2/Re*sym(grad(v))) * my_ds
                 + 1/Re*(sigma/h)*inner(v, u-u_ex) * my_ds
                 + advect * dot(v, uflux_ext_2) * my_ds
            )

            if bcs_ids_dont_apply is not None:
                F_DG += (
                    - inner(outer(u, n), 2/Re*sym(grad(v))) * ds(bcs_ids_dont_apply)
                    + 1/Re*(sigma/h)*inner(v, u) * ds(bcs_ids_dont_apply)
                   )

        F += F_DG

    elif discr == "cg":
        F += advect * inner(dot(grad(u), u), v) * dx

    if rhs:
        F -= inner(B_rhs, C)*dx
        
    return F

def J_form(F, z, test_z):
    (u, p, B, E) = split(z)
    (v, q, C, Ff) = split(test_z)
    w = TrialFunction(Z)
    [w_u, w_p, w_B, w_E] = split(w)

    J_newton = ufl.algorithms.expand_derivatives(derivative(F, z, w))

    if linearisation == "newton":
        J = J_newton

    elif linearisation == "mdp":
        J_mdp = (
              J_newton
            - dt_factor * dt * inner(scross(w_u, B), Ff) * dx  # G
                    )
        J = J_mdp

    elif linearisation == "picard":
        J_picard = (
              J_newton
            - dt_factor * dt * S * inner(vcross(w_B, E), v) * dx  # J_tilde
            - dt_factor * dt * S * inner(vcross(w_B, scross(u, B)), v) * dx  # D_1_tilde
            - dt_factor * dt * S * inner(vcross(B, scross(u, w_B)), v) * dx  # D_2_tilde
            - dt_factor * dt * inner(scross(u, w_B), Ff) * dx  # G_tilde
                    )
        J = J_picard

    else:
        raise ValueError("only know newton, mdp and picard as linearisation method")

    return J

def initial_condition():
    z = Function(Z)
    t.assign(0)
    z.sub(0).interpolate(u_ex)
    z.sub(1).interpolate(p_ex)
    if testproblem == "island_coal":
        z.sub(2).interpolate(B_ex + B_pert)
    else:    
        z.sub(2).interpolate(B_ex)
    z.sub(3).interpolate(E_ex)

    return z


# Initial Condition
z0 = initial_condition()
z1 = initial_condition()
z = Function(z0)
z_test = TestFunction(Z)

dt_factor = Constant(1)

# Crank Nicolson form
F_cn = (
     inner(split(z)[0], split(z_test)[0])*dx
   - inner(split(z0)[0], split(z_test)[0])*dx
   + inner(split(z)[2], split(z_test)[2])*dx
   - inner(split(z0)[2], split(z_test)[2])*dx
   + 0.5*dt*(form(z, z_test, Z, 0) + form(z0, z_test, Z, 1))
  )

# Implicit Euler form
F_ie = (
     inner(split(z)[0], split(z_test)[0])*dx
   - inner(split(z0)[0], split(z_test)[0])*dx
   + inner(split(z)[2], split(z_test)[2])*dx
   - inner(split(z0)[2], split(z_test)[2])*dx
   + dt*(form(z, z_test, Z, 0))
  )

# BDF2 form
F_bdf2 = (
     inner(split(z)[0], split(z_test)[0])*dx
   - 4.0/3.0*inner(split(z0)[0], split(z_test)[0])*dx
   + 1.0/3.0*inner(split(z1)[0], split(z_test)[0])*dx
   + inner(split(z)[2], split(z_test)[2])*dx
   - 4.0/3.0*inner(split(z0)[2], split(z_test)[2])*dx
   + 1.0/3.0*inner(split(z1)[2], split(z_test)[2])*dx
   + 2.0/3.0*dt*form(z, z_test, Z, 0)
  )

JspaceCG = FunctionSpace(mesh, "CG", 1)

# functions to measure J
j_CG = Function(JspaceCG)
testCG = TestFunction(JspaceCG)
trialCG = TrialFunction(JspaceCG)

(u_, p_, B_, E_) = z.split()
lhs_BCG = inner(trialCG,testCG)*dx
rhs_BCG = inner(B_,vcurl(testCG))*dx

postprocessing_params_CG = {
                          "snes_type": "ksponly",
                          "snes_lag_jacobian": -2,
                          "ksp_type": "cg",
                          "pc_type": "ml",
                         }

problem_BCG = LinearVariationalProblem(lhs_BCG, rhs_BCG, j_CG, constant_jacobian=True)
solver_BCG = LinearVariationalSolver(problem_BCG, solver_parameters=postprocessing_params_CG)
solver_BCG.solve()
j00CG = j_CG((0,0))

if mesh.comm.rank == 0:
    print("At midpoint, j00 is %s" % j00CG)

outputdir = "./results/"
Nx = int(float(baseN)*2**float(nref))
filename = "reconrate_{0:}_{1:}_{2:}x{3:}.txt".format(int(float(Rem)),int(float(Re)), Nx, Nx)
reconrateCG = open(outputdir + filename, "w")

if stab:
    initial = interpolate(as_vector([sin(y), x]), V)
    z_last_u.assign(initial)
    stabilisation = BurmanStabilisation(Z.sub(0), state=z_last_u, h=FacetArea(mesh), weight=stab_weight)
    stabilisation_form_u = stabilisation.form(split(z)[0], split(z_test)[0])
    F_ie += dt*(advect * stabilisation_form_u)
    F_cn += 0.5*dt*(advect * stabilisation_form_u)
    F_bdf2 += 2.0/3.0*dt*(advect * stabilisation_form_u)

J_cn = J_form(F_cn, z, z_test)
J_bdf2 = J_form(F_bdf2, z, z_test)
J_ie = J_form(F_ie, z, z_test)

appctx = {"Re": Re, "gamma": gamma, "nu": 1/Re, "Rem": Rem, "gamma2": gamma2}
params = solvers[args.solver_type]

# Depending on the Mesh Hierarchy we have to use star or macrostar solver
if args.solver_type in ["fs2by2", "fs2by2slu"]:
    params["fieldsplit_0"] = nsfsstar if hierarchy == "uniform" else nsfsmacrostar

if args.solver_type in ["fs2by2", "fs2by2nslu"]:
    params["fieldsplit_1"] = outerschurstar if hierarchy == "uniform" else outerschurmacrostar

# Set up nonlinear solver for first time step
nvproblem_cn = NonlinearVariationalProblem(F_cn, z, bcs=bcs, J=J_cn)
solver_cn = NonlinearVariationalSolver(nvproblem_cn, solver_parameters=params, options_prefix="", appctx=appctx)

# Set up nonlinear solver for first time step
nvproblem_ie = NonlinearVariationalProblem(F_ie, z, bcs=bcs, J=J_ie)
solver_ie = NonlinearVariationalSolver(nvproblem_ie, solver_parameters=params, options_prefix="", appctx=appctx)

# Set up nonlinear solver for later time steps
nvproblem_bdf2 = NonlinearVariationalProblem(F_bdf2, z, bcs=bcs, J=J_bdf2)
solver_bdf2 = NonlinearVariationalSolver(nvproblem_bdf2, solver_parameters=params, options_prefix="", appctx=appctx)

# Definition of solver and transfer operators
qtransfer = NullTransfer()
Etransfer = NullTransfer()
vtransfer = SVSchoeberlTransfer((1/Re, gamma), 2, hierarchy)
Btransfer = SVSchoeberlTransfer((1/Rem, gamma2), 2, hierarchy)
dgtransfer = DGInjection()

transfers = {
                Q.ufl_element(): (prolong, restrict, qtransfer.inject),
                R.ufl_element(): (prolong, restrict, Etransfer.inject),
                VectorElement("DG", mesh.ufl_cell(), args.k): (dgtransfer.prolong, restrict, dgtransfer.inject),
            }

# On barycentric refined grids we need special prolongation operators
if hierarchy == "bary":
    transfers[V.ufl_element()] = (vtransfer.prolong, vtransfer.restrict, inject)

transfermanager = TransferManager(native_transfers=transfers)
solver_cn.set_transfer_manager(transfermanager)
solver_bdf2.set_transfer_manager(transfermanager)

results = {}
res = args.Re
rems = args.Rem
Ss = args.S

pvd = File("output/mhd.pvd")
pvd_init = File("output/init.pvd")

if args.output:
    pvd_init.write(z.sub(0), z.sub(1), z.sub(2), z.sub(3))

def run(re, rem, s):
    (u, p, B, E) = z.split()
    Re.assign(re)
    Rem.assign(rem)
    S.assign(s)

    z0.assign(initial_condition())
    z1.assign(initial_condition())
    z.assign(z0)

    # Indices for output depending on Re-S, Re-Rem or S-Rem table
    if len(args.S) == 1 or len(args.Rem) == 1:
        ind1 = re
        ind2 = s*rem
    else:
        ind1 = re*s
        ind2 = rem

    # Set things up for timestepping
    T = args.Tf  # final time
    t.assign(0.0)  # current time we are solving for
    global dt
    global bcs
    ntimestep = 0  # number of timesteps solved
    total_nonlinear_its = 0  # number of total nonlinear iterations
    total_linear_its = 0  # number of total linear iterations
    tot_time_from_3 = 0
    tot_nonlin_its_from_3 = 0
    tot_lin_its_from_3 = 0

    while (float(t) < float(T-dt)+1.0e-10):
        t.assign(t+dt)
        if mesh.comm.rank == 0:
            print(BLUE % ("\nSolving for time: %f" % t), flush=True)

        if bc_varying:
            if not solution_known:
                raise ValueError("Sorry, don't know how to reconstruct the BCs")
            else:
                u_ex_ = interpolate(u_ex, V)
                B_ex_ = interpolate(B_ex, W)
                p_ex_ = interpolate(p_ex, Q)
                E_ex_ = interpolate(E_ex, R)
                bcs[0].function_arg = u_ex_
                bcs[1].function_arg = B_ex_
                bcs[2].function_arg = E_ex_

        if mesh.comm.rank == 0:
            print(GREEN % ("Solving for #dofs = %s, Re = %s, Rem = %s, gamma = %s, gamma2 = %s, S = %s, baseN = %s, nref = %s, "
                           "linearisation = %s, testproblem = %s, discr = %s, k = %s"
                           % (Z.dim(), float(re), float(rem), float(gamma), float(gamma2), float(S), int(baseN), int(nref),
                              linearisation, testproblem, discr, int(float(k)))), flush=True)
        if not os.path.exists("results/"):
            os.mkdir("results/")

        # Update z_last_u in Burman Stabilisation
        if stab:
            stabilisation.update(z.split()[0])
            z_last_u.assign(u)

        # Do CN in first timestep, after that BDF2
        start = datetime.now()
        if ntimestep < 1:
            dt_factor.assign(0.5)
            solver = solver_cn
            solver_cn.solve()
        else:
            dt_factor.assign(2.0/3.0)
            solver = solver_bdf2
            solver_bdf2.solve()
        end = datetime.now()

        # Iteration numbers for this time step
        linear_its = solver.snes.getLinearSolveIterations()
        nonlinear_its = solver.snes.getIterationNumber()
        time = (end-start).total_seconds() / 60

        if nonlinear_its == 0:
            nonlinear_its = 1

        if linear_its == 0:
            linear_its = 1

        if mesh.comm.rank == 0:
            print(GREEN % ("Time taken: %.2f min in %d nonlinear iterations, %d linear iterations (%.2f Krylov iters per Newton step)"
                           % (time, nonlinear_its, linear_its, linear_its/float(nonlinear_its))), flush=True)
            print("%.2f @ %d @ %d @ %.2f" % (time, nonlinear_its, linear_its, linear_its/float(nonlinear_its)), flush=True)

        (u, p, B, E) = z.split()

        pintegral = assemble(p*dx)
        p.assign(p - Constant(pintegral/area))

        B.rename("MagneticField")
        u.rename("VelocityField")
        p.rename("Pressure")
        E.rename("ElectricFieldf")

        norm_div_u = sqrt(assemble(inner(div(u), div(u))*dx))
        norm_div_B = sqrt(assemble(inner(div(B), div(B))*dx))

        if mesh.comm.rank == 0:
            print("||div(u)||_L^2 = %s" % norm_div_u, flush=True)
            print("||div(B)||_L^2 = %s" % norm_div_B, flush=True)

        if solution_known:
            B_ex_ = interpolate(B_ex, B.function_space())
            u_ex_ = interpolate(u_ex, u.function_space())
            p_ex_ = interpolate(p_ex, p.function_space())
            E_ex_ = interpolate(E_ex, E.function_space())
            B_ex_.rename("ExactSolutionB")
            u_ex_.rename("ExactSolutionu")
            p_ex_.rename("ExactSolutionp")
            E_ex_.rename("ExactSolutionE")

            # Compute errors for MMS
            error_u = errornorm(u_ex, u, 'L2')
            error_B = errornorm(B_ex, B, 'L2')
            error_E = errornorm(E_ex, E, 'L2')
            error_p = errornorm(p_ex, p, 'L2')

            if mesh.comm.rank == 0:
                print("Error ||u_ex - u||_L^2 = %s" % error_u, flush=True)
                print("Error ||p_ex - p||_L^2 = %s" % error_p, flush=True)
                print("Error ||B_ex - B||_L^2 = %s" % error_B, flush=True)
                print("Error ||E_ex - E||_L^2 = %s" % error_E, flush=True)

                f = open("error.txt", 'a+')
                f.write("%s,%s,%s,%s\n" % (error_u, error_p, error_B, error_E))
                f.close()

            if output:
                pvd.write(u, u_ex_, p, p_ex_, B, B_ex_, E, E_ex_, time=float(t))

            sys.stdout.flush()
            info_dict = {
                "Re": re,
                "Rem": rem,
                "S": s,
                "krylov/nonlin": linear_its/nonlinear_its,
                "nonlinear_iter": nonlinear_its,
                "error_u": error_u,
                "error_p": error_p,
                "error_B": error_B,
                "error_E": error_E,
            }

        else:
            info_dict = {
                "Re": re,
                "Rem": rem,
                "S": s,
                "krylov/nonlin": linear_its/nonlinear_its,
                "nonlinear_iter": nonlinear_its,
            }

            if output:
                pvd.write(u, p, B, E, time=float(t))

        message(BLUE % info_dict)

        z_last_u.assign(u)
        if not os.path.exists("dump/"):
            os.mkdir("dump/")

        # Compute current density and compute reconnection rate, output to screen and file
        (u_, p_, B_, E_) = z.split()

        solver_BCG.solve()
        # Check at the origin, CG
        ReconRate = (1/float(Rem))*(j_CG((0,0)) - j00CG)

        if mesh.comm.rank == 0:
            print("Reconnection Rate (CG) at midpoint is %e" % ReconRate)
        reconrateCG.write("%e&%e\n" % (t,ReconRate))
        reconrateCG.flush()
        sys.stdout.flush()
        
        # update time step
        z1.assign(z0)
        z0.assign(z)
        if ntimestep >=2:
           tot_time_from_3 += time
           tot_nonlin_its_from_3 += nonlinear_its
           tot_lin_its_from_3 += linear_its
        ntimestep += 1
        total_nonlinear_its += nonlinear_its
        total_linear_its += linear_its

    # Calculate average number of nonlinear iterations per timestep and linear iterations per nonlinear iteration
    avg_nonlinear_its = total_nonlinear_its / float(ntimestep)
    avg_linear_its = total_linear_its / float(total_nonlinear_its)

    # Write iteration numbers to file
    if mesh.comm.rank == 0:
        dir = 'results/results'+str(linearisation)+str(testproblem)+'/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        f = open(dir+str(float(ind1))+str(float(ind2))+'.txt', 'w+')
        f.write("({0:2.1f}){1:4.1f}".format(float(avg_nonlinear_its), float(avg_linear_its)))
        f.close()

        print(GREEN % ("Average %2.1f nonlinear iterations, %2.1f linear iterations"
                       % (avg_nonlinear_its, avg_linear_its)), flush=True)

        with open('results/scaling.txt', 'a') as fd:
            time_per_it = float(time)/float(linear_its)
            outstr = f"baseN={int(baseN)}, nref={int(nref)}, Ncores={int(MPI.COMM_WORLD.size):2.0f}, time/linear_it={time_per_it:0.2f}"
            fd.write(outstr)
            fd.write("\n")

        # Save runtimes from third timestep on
        dir = 'results/times'+str(float(baseN)*2**float(nref))+str(linearisation)+str(testproblem)+'/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        f = open(dir+str(float(ind1))+str(float(ind2))+'.txt', 'w+')
        f.write("{0:.2f} {1:.2f}".format(float(tot_time_from_3), float(tot_time_from_3)/float(tot_lin_its_from_3)))
        f.close()

# Loop over parameters
for rem in rems:
    for s in Ss:
        for re in res:
            run(re, rem, s)
