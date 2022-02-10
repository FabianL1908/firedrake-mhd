from firedrake import *
from datetime import datetime

class MHDSolver(object):

    def __init__(self, args, problem):
        self.problem = problem
        self.args = args
        self.z = problem.z
        self.mesh = problem.mesh

    def get_solver_params(self):
        return NotImplementedError

    def get_solver(self, appctx):
        params = self.get_solver_params()
        F = self.problem.form()
        bcs = self.problem.bcs(self.problem.Z)[0]
        J = self.problem.jacobian(F)
        my_problem = NonlinearVariationalProblem(F, self.z, bcs, J=J)
        solver = NonlinearVariationalSolver(my_problem, solver_parameters=params, options_prefix="", appctx=appctx)
        transfers = self.problem.transfer_ops()
        transfermanager = TransferManager(native_transfers=transfers)
        solver.set_transfer_manager(transfermanager)
        return solver

    def solve(self, *args):
        raise NotImplementedError

class StandardMHDSolver(MHDSolver):

    def __init__(self, args, problem):
        super().__init__(args, problem)
        self.Re = problem.Re
        self.Rem = problem.Rem
        self.S = problem.S
        self.problem = problem
        self.args = args

    def get_solver_params(self):
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
            "mat_mumps_icntl_14": 5000,
        }

        return lu

    def solve(self, *args):
        re = args[0]
        rem = args[1]
        s = args[2]

        self.Re.assign(re)
        self.Rem.assign(rem)
        self.S.assign(s)

        appctx = {"Re": self.Re, "gamma": self.args.gamma, "nu": 1/self.Re}
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
