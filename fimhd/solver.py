from firedrake import *
from datetime import datetime
from mpi4py import MPI
from fimhd.utils import *
import petsc4py
petsc4py.PETSc.Sys.popErrorHandler()

class MHDSolver(object):

    def __init__(self, args, problem):
        self.problem = problem
        self.args = args
        self.z = problem.z
        self.mesh = problem.mesh

    def get_transfer_manager(self):
        transfers = self.problem.transfer_ops()
        return TransferManager(native_transfers=transfers)

        
    def get_solver(self, appctx):
        params = self.get_solver_params()
        F = self.problem.form()
        bcs = self.problem.bcs(self.problem.Z)[0]
        J = self.problem.jacobian(F)
        my_problem = NonlinearVariationalProblem(F, self.z, bcs, J=J)
        solver = NonlinearVariationalSolver(my_problem, solver_parameters=params, options_prefix="", appctx=appctx)
        transfer_manager = self.get_transfer_manager()
        solver.set_transfer_manager(transfer_manager)
        return solver
    
    def solve(self, *args):
        raise NotImplementedError

    def get_solver_params(self):
        solvers = self.get_solver_dict()
        return solvers[self.args.solver_type]

    def get_solver_dict(self):
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

        nsfs = {
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
        outerschur = {
            "ksp_type": "fgmres",
            "ksp_atol": 1.0e-7,
            "ksp_rtol": 1.0e-7,
            "pc_type": "python",
#            "pc_python_type": __name__ + ".SchurPCBE",
#            "aux_mg_transfer_manager": __name__ + ".transfermanager",
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
#            "pc_python_type": __name__ + ".SchurPCBE",
            "aux_pc_type": "lu",
            "aux_pc_factor_mat_solver_type": "mumps",
            "aux_mat_mumps_icntl_14": ICNTL_14,
           }
        
        if self.args.hierarchy == "uniform":
            if self.args.tinyasm:
                mg_levels_pc_dict = {
                    "fieldsplit_0_mg_levels_pc_star_backend": "tinyasm"
                }
            else:
                mg_levels_pc_dict = {
                    "fieldsplit_0_mg_levels_pc_star_construct_dim": 0,
                    "fieldsplit_0_mg_levels_pc_star_sub_sub_ksp_type": "preonly",
                    "fieldsplit_0_mg_levels_pc_star_sub_sub_pc_type": "lu",
                    "fieldsplit_0_mg_levels_pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack"
                }
        elif self.args.hierarchy == "bary":
            mg_levels_pc_dict = {
                "fieldsplit_0_mg_levels_pc_python_type": "firedrake.PatchPC",
                "fieldsplit_0_mg_levels_patch_pc_patch_save_operators": True,
                "fieldsplit_0_mg_levels_patch_pc_patch_partition_of_unity": False,
                "fieldsplit_0_mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
                "fieldsplit_0_mg_levels_patch_pc_patch_construct_dim": 0,
                "fieldsplit_0_mg_levels_patch_pc_patch_construct_type": "python",
                "fieldsplit_0_mg_levels_patch_pc_patch_construct_python_type": "alfi.MacroStar",
                "fieldsplit_0_mg_levels_patch_sub_ksp_type": "preonly",
                "fieldsplit_0_mg_levels_patch_sub_pc_type": "lu",
                "fieldsplit_0_mg_levels_patch_sub_pc_factor_mat_solver_type": "umfpack",
                }

        nsfs = {**nsfs, **mg_levels_pc_dict}
        outerschur = {**outerschur, **mg_levels_pc_dict}

        outerbase = {
            "snes_type": "newtonls",
            "snes_max_it": 10,
            "snes_linesearch_type": "basic",
            "snes_linesearch_maxstep": 1.0,
            "snes_rtol": 1.0e-10,
            "snes_atol": 4.0e-6,
            "snes_monitor": None,
            "ksp_type": "fgmres",
            "ksp_max_it": 35,
            "ksp_atol": 3.0e-6,
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

        nsfs, outerschur, outerschurlu = self.configure_solver(nsfs, outerschur, outerschurlu)

        fs2by2_dict = {
            "fieldsplit_0": nsfs,
            "fieldsplit_1": outerschur
            }
        fs2by2 = {**outerbase, **fs2by2_dict}

        # Main solver with LU solver for (u,p)-block        
        fs2by2nslu_dict = {
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "lu",
            "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
            "fieldsplit_0_mat_mumps_icntl_14": ICNTL_14,
            "fieldsplit_1": outerschur
            }
        fs2by2nslu = {**outerbase, **fs2by2nslu_dict}

        # Main solver with LU solver for Schur complement
        fs2by2slu_dict = {
            "fieldsplit_0": nsfs,
            "fieldsplit_1": outerschurlu 
            }
        fs2by2slu = {**outerbase, **fs2by2slu_dict}

        # Main solver with LU solver for (E,B)-block and Schur complement
        fs2by2lu_dict = {
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "lu",
            "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
            "fieldsplit_0_mat_mumps_icntl_14": ICNTL_14,
            "fieldsplit_1": outerschurlu 
            }
        fs2by2lu = {**outerbase, **fs2by2lu_dict}

        solvers = {"lu": lu, "fs2by2": fs2by2, "fs2by2nslu": fs2by2nslu,
                   "fs2by2slu": fs2by2slu, "fs2by2lu": fs2by2lu}

        return solvers


    def configure_solver(self, *args):
        raise NotImplementedError

class SchurPCBE(AuxiliaryOperatorPC):
    def form(self, pc, V, U):
        [B, E] = split(U)
        [C, Ff] = split(V)
        state = self.get_appctx(pc)['state']
        [u_n, p_n, B_n, E_n] = split(state)
        Rem = self.get_appctx(pc)['Rem']

        A = (
             + 1*inner(E, Ff) * dx
             + inner(scross(u_n, B), Ff) * dx
            - 1/Rem * inner(B, vcurl(Ff)) * dx
             + inner(vcurl(E), C) * dx
            + 1/Rem * inner(div(B), div(C)) * dx
                      )

        bcs = [DirichletBC(V.function_space().sub(0), 0, "on_boundary"),
               DirichletBC(V.function_space().sub(1), 0, "on_boundary"),
               ]

        return (A, bcs)

class StandardMHDSolver(MHDSolver):

    def __init__(self, args, problem):
        super().__init__(args, problem)
        self.Re = problem.Re
        self.Rem = problem.Rem
        self.S = problem.S
        self.problem = problem
        self.args = args


    def configure_solver(self, nsfs, outerschur, outerschurlu):
        global my_tmanager
        my_tmanager = self.get_transfer_manager()
        schur_dict = {
            "pc_python_type": __name__ + ".SchurPCBE",
            "aux_mg_transfer_manager": __name__ + ".my_tmanager",
        }
        outerschur = {**outerschur, **schur_dict}
        outerschurlu = {**outerschurlu, **schur_dict}
        return nsfs, outerschur, outerschurlu
        
    def solve(self, *args):
        re = args[0]
        rem = args[1]
        s = args[2]

        self.Re.assign(re)
        self.Rem.assign(rem)
        self.S.assign(s)

        appctx = {"Re": self.Re, "gamma": self.args.gamma, "nu": 1/self.Re, "Rem": self.Rem}
        solver = self.get_solver(appctx)

        if self.problem.mesh.comm.rank == 0:
            print(GREEN % ("Solving for #dofs = %s, Re = %s, Rem = %s, gamma = %s, S = %s, baseN = %s, nref = %s, "
                           "linearisation = %s, k = %s"
                           % (self.problem.Z.dim(), float(re), float(rem), float(self.problem.gamma), float(s), int(self.args.baseN), int(self.problem.nref),
                              self.problem.linearisation, int(float(self.problem.k)))), flush=True)

        start = datetime.now()
        solver.solve()
        end = datetime.now()

        linear_its = solver.snes.getLinearSolveIterations()
        nonlinear_its = solver.snes.getIterationNumber()
        time = (end-start).total_seconds() / 60

        if self.problem.mesh.comm.rank == 0:
            if nonlinear_its == 0:
                nonlinear_its = 1
            print(GREEN % ("Time taken: %.2f min in %d nonlinear iterations, %d linear iterations (%.2f Krylov iters per Newton step)"
                           % (time, nonlinear_its, linear_its, linear_its/float(nonlinear_its))), flush=True)
            print("%.2f @ %d @ %d @ %.2f" % (time, nonlinear_its, linear_its, linear_its/float(nonlinear_its)), flush=True)

        (u, p, B, E) = self.z.split()
        
        # Make sure that average of p is 0
        pintegral = assemble(p*dx)
        area = assemble(Constant(1, domain=self.problem.mesh)*dx)
        p.assign(p - Constant(pintegral/area))

        B.rename("MagneticField")
        u.rename("VelocityField")
        p.rename("Pressure")
        E.rename("ElectricFieldf")

        # Compute divergence of u and B
        norm_div_u = sqrt(assemble(inner(div(u), div(u))*dx))
        norm_div_B = sqrt(assemble(inner(div(B), div(B))*dx))

        if self.mesh.comm.rank == 0:
            print("||div(u)||_L^2 = %s" % norm_div_u, flush=True)
            print("||div(B)||_L^2 = %s" % norm_div_B, flush=True)


class HallMHDSolver(MHDSolver):
    pass


class BoussinesqSolver(MHDSolver):
    pass
