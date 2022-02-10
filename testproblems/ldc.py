from firedrake import *
from fimhd import *

class LidDrivenCavityStandardProblem(StandardMHDProblem):
    def __init__(self, args):
        super().__init__(args)

    def base_mesh(self, distribution_parameters):
        if self.args.dim == 2:
            base = UnitSquareMesh(self.args.baseN, self.args.baseN, diagonal="crossed",
                                  distribution_parameters=distribution_parameters)
            base.coordinates.dat.data[:, 0] -= 0.5
            base.coordinates.dat.data[:, 1] -= 0.5
        elif self.args.dim == 3:
            pass

        return base

    def bcs(self, Z):
        mesh = Z.mesh()
        if self.args.dim == 2:
            u_ex = Constant((1, 0), domain=mesh)
            B_ex = Constant((0, 1), domain=mesh)
            B_ex = project(B_ex, Z.sub(2))
            p_ex = E_ex = Constant(0, domain=mesh)

            bcs = [DirichletBC(Z.sub(0), u_ex, 4),  # 4 == upper boundary (y==1)
                   DirichletBC(Z.sub(0), 0, (1, 2, 3)),
                   DirichletBC(Z.sub(2), B_ex, "on_boundary"),
                   DirichletBC(Z.sub(3), 0, "on_boundary"),
                   PressureFixBC(Z.sub(1), 0, 1)]
            bcs_ids_apply = 4
            bcs_ids_dont_apply = (1, 2, 3)
            sol_ex = (u_ex, p_ex, B_ex, E_ex)

        elif self.args.dim == 3:
            pass

        return bcs, bcs_ids_apply, bcs_ids_dont_apply, sol_ex

    def rhs(self, Z):
        return None

if __name__ == "__main__":

    parser = get_default_parser()
    args, _ = parser.parse_known_args()
    problem = LidDrivenCavityStandardProblem(args)
    solver = StandardMHDSolver(args, problem)

    run_solver(solver, args.Re, args.Rem, args.S)

    
