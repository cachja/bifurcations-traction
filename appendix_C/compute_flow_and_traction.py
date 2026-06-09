from firedrake import *
from petsc4py import PETSc
import numpy as np
import os

_print = print
def print(x):
    if COMM_WORLD.rank == 0:
        _print(x, flush=True)


def create_new_mesh(Len,Wid,Obs_x,Obs_y,Rad,Var,hmax,hmax_loc):
    import firedrake as fd
    from netgen.geom2d import SplineGeometry
    import netgen

    if Var <= 1e-8:
        if fd.COMM_WORLD.rank == 0:
            geo = SplineGeometry()
            geo.AddRectangle((0, 0), (Len, Wid), bcs=(2, 3, 2, 1))
            geo.AddCircle((Obs_x, Obs_y), r=Rad, leftdomain=0,
                        rightdomain=1, bc=5, maxh=hmax_loc)
            ngmesh = geo.GenerateMesh(maxh=hmax)

            # Refine manually (near the outflow important for higher Re when vortices meets the outflow)
            for element in ngmesh.Elements2D():
                nodes = [ngmesh[ep] for ep in element.points]

                # --- 1. inflow corner refinement ---
                if any(
                    (n.p[0] < 0.01 and (n.p[1] < 0.01 or n.p[1] > Wid - 0.01))
                    for n in nodes
                ):
                    element.refine = True

                elif any(
                    (n.p[0] > Len-0.01 and (n.p[1] < 0.01 or n.p[1] > Wid - 0.01))
                    for n in nodes
                ):
                    element.refine = True

                # --- 2. wake refinement (downstream region) ---
                elif any(
                    (n.p[0] > Obs_x + Rad) and (n.p[0] < Obs_x + 10*Rad) and
                    (abs(n.p[1] - Obs_y) < 2 * Rad)
                    for n in nodes
                ):
                    element.refine = True

                else:
                    element.refine = False
            ngmesh.Refine(adaptive=True)
            # # Full red-green refinement
            # ngmesh.Refine()
            # # Full barycentric refinement
            # ngmesh.SplitAlfeld()
        else:
            ngmesh = netgen.libngpy._meshing.Mesh(2)
    else:
        if fd.COMM_WORLD.rank == 0:
            geo = SplineGeometry()
            geo.AddRectangle((0, 0), (Len, Wid), bcs=(2, 3, 2, 1))
            npts = int(2*3.14*Rad/hmax_loc)
            thetas = np.linspace(0, 2*np.pi, npts, endpoint=False)
            petals = 8
            points = []
            for th in thetas:
                r = Rad * (1 + Var * np.sin(petals * th))
                x = Obs_x + r * np.cos(th)
                y = Obs_y + r * np.sin(th)
                points.append((x, y))

            # close curve
            pnums = [geo.AddPoint(float(p[0]), float(p[1]), maxh=hmax_loc) for p in points]

            
            curves = []
            n = len(pnums)

            for i in range(n):
                p1 = pnums[i]
                p2 = pnums[(i+1) % n]

                curves.append([["line", p1, p2], 5])  # bc=5
            for c, bc in curves:
                geo.Append(c, bc=bc,leftdomain=0, rightdomain=1)

            ngmesh = geo.GenerateMesh(maxh=hmax)

            # Refine manually (near the outflow important for higher Re when vortices meets the outflow)
            for element in ngmesh.Elements2D():
                nodes = [ngmesh[ep] for ep in element.points]

                # --- 1. inflow corner refinement ---
                if any(
                    (n.p[0] < 0.01 and (n.p[1] < 0.01 or n.p[1] > Wid - 0.01))
                    for n in nodes
                ):
                    element.refine = True

                # --- 2. wake refinement (downstream region) ---
                elif any(
                    (n.p[0] > Obs_x + Rad) and (n.p[0] < Obs_x + 10*Rad) and
                    (abs(n.p[1] - Obs_y) < 2 * Rad)
                    for n in nodes
                ):
                    element.refine = True

                else:
                    element.refine = False

            ngmesh.Refine(adaptive=True)
        else:
            ngmesh = netgen.libngpy._meshing.Mesh(2)
    return fd.Mesh(ngmesh, comm=fd.COMM_WORLD)


info = PETSc.Sys.Print
logging.set_log_level(INFO) #DEBUG, INFO, WARNING, ERROR, CRITICAL

def a(v,u,nu=Constant(0.001)):
    return (inner(grad(v)*v, u) + inner(nu*grad(v), grad(u)))*dx

def b_form(q,v):
    return inner(div(v),q)*dx

def F1(v,p,v_):
    return a(v,v_) - b_form(p,v_)

def build_solver(W,w,mesh,Umax,Wid):

    x = SpatialCoordinate(mesh)
    Umax_const = Constant(Umax)
    inflow_profile = as_vector([4.0*Umax_const*x[1]*(Wid-x[1])/Wid**2,0])

    bc_inlet = DirichletBC(W.sub(0), inflow_profile, 1)
    bc_cylinder = DirichletBC(W.sub(0), (0, 0), 5)
    bc_walls = DirichletBC(W.sub(0), (0, 0), 2)
    bc_outlet = DirichletBC(W.sub(0).sub(1), 0, 3)
    bcs = [bc_inlet, bc_cylinder, bc_walls, bc_outlet]
    # Define unknown and test function(s)
    (v_, p_) = TestFunctions(W)

    # current unknown time step
    (v, p) = split(w)

    F = a(v,v_) - b_form(p_,v) - b_form(p,v_)

    J = derivative(F, w)

    problem=NonlinearVariationalProblem(F,w,bcs,J)
    lu = {"mat_type": "aij",
          "snes_type": "newtonls",
          #"snes_monitor": None,
          "snes_converged_reason": None,
          "snes_max_it": 200,
          "snes_rtol": 1e-11,
          "snes_atol": 5e-10,
          "snes_linesearch_type": "basic",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
            # Reuse Jacobian --- good tactics but not robust implementation
            "snes_lag_jacobian": 5,
            "snes_lag_jacobian_persists": True,
            "snes_lag_preconditioner": 5,  # reuse preconditioner too
            "snes_lag_preconditioner_persists": True,
            # Reuse Jacobian end
            }
    solver = NonlinearVariationalSolver(problem, solver_parameters=lu)
    return solver, Umax_const

def build_traction_solver_old(V,w,gg):
    v, p  = w.subfunctions
    g = TrialFunction(V)
    g_ = TestFunction(V)
    # set BC
    bc_list = [1,2,3] # zero traction on bndries not touching bndry of interest
    bc_num = 5 # circle bndry - here we solve PS
    bcs_g = [DirichletBC(V, Constant((0,0)), i) for i in bc_list]

    # Define the bilinear form for Poincare-Steklov problem
    a_g = inner(g , g_) * ds(bc_num)
    L_g = F1(v,p,g_)

    # Solve linear system
    A = assemble(a_g, bcs = bcs_g)
    b = assemble(L_g)

    # Dirty ident_zeros trick
    diagonal = A.petscmat.getDiagonal()
    vals = diagonal.array
    vals[vals == 0] = 1
    A.petscmat.setDiagonal(diagonal, PETSc.InsertMode.INSERT_VALUES)

    problem = LinearVariationalProblem(A,b,gg)
    solver_traction = LinearVariationalSolver(problem)

    return solver_traction


def build_traction_solver(V, w, gg):
    v, p = w.subfunctions

    g  = TrialFunction(V)
    g_ = TestFunction(V)

    bc_list = [1, 2, 3]
    bc_num = 5
    bcs_g = [DirichletBC(V, Constant((0,0)), i) for i in bc_list]

    a_g = inner(g, g_) * ds(bc_num) + 1e-12*inner(g,g_)*dx
    L_g = F1(v, p, g_)   # depends on w → will be reassembled

    problem = LinearVariationalProblem(a_g, L_g, gg, bcs=bcs_g)

    solver_params = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_type": "aij",
    }

    solver = LinearVariationalSolver(problem,
        solver_parameters=solver_params)

    return solver

    
def compute_traction_bndryval(V,gg):
    import boundary_values
    (drag_coords, drag_values) = boundary_values.get_boundary_values(V,mesh,5,gg,0)
    drag_values = np.array(drag_values).tolist()
    (lift_coords, lift_values) = boundary_values.get_boundary_values(V,mesh,5,gg,1)
    lift_values = np.array(lift_values).tolist()

    return (drag_coords,drag_values,lift_coords,lift_values)

if __name__ == "__main__":

    Len = 2.2
    Rad = 0.05
    Wid = 0.41
    Obs_x = 0.2

    problem_list = ["cylinder","cylinder_symm","flower","flower_symm","cylinder_symm_wide","cylinder_wide"]
    problem = problem_list[4]
    print(f"{problem=}")
    if problem == "cylinder":
        data_dir = "./output_turek_cylinder/"
        Var = 0.0
        Obs_y = 0.2
        hmax = Wid/25/2

    elif problem == "cylinder_symm":
        data_dir = "./output_turek_cylinder_symm/"
        Var = 0.0
        Obs_y = Wid/2
        hmax = Wid/25/2

    elif problem == "flower":
        data_dir = "./output_turek_flower/"
        Var = Rad/5
        Obs_y = 0.2
        hmax = Wid/25/4

    elif problem == "flower_symm":
        data_dir = "./output_turek_flower_symm/"
        Var = Rad/5
        Obs_y = Wid/2
        hmax = Wid/25/4

    elif problem == "cylinder_symm_wide":
        data_dir = "./output_turek_cylinder_symm_wide/"
        Var = 0.0
        Wid = Wid*2
        Obs_y = Wid/2
        hmax = Wid/25/8

    elif problem == "cylinder_wide":
        data_dir = "./output_turek_cylinder_wide/"
        Var = 0.0
        Wid = Wid*2
        Obs_y = Wid*0.487805
        hmax = Wid/25/8

    else:
        raise ValueError(f"Unknown problem: {problem}")
 
    hmax_loc = hmax/10

    mesh = create_new_mesh(Len,Wid,Obs_x,Obs_y,Rad,Var,hmax,hmax_loc)

    # Build spaces for traction (linear problem)
    Vspace = VectorFunctionSpace(mesh,"CG",2)
    Pspace = FunctionSpace(mesh,"CG",1)
    GGspace = VectorFunctionSpace(mesh,"CG",1)
    W = Vspace*Pspace
    info(f"{W.dim()=}")
    info(f"{GGspace.dim()=}")
    w = Function(W)
    gg = Function(GGspace)

    solver, Umax_const = build_solver(W,w,mesh,0,Wid)
    solver_traction = build_traction_solver(GGspace,w,gg)

    Re_list = np.linspace(0,501,1003)
    Umax_list = np.array(Re_list)*3/2/100

    
    lift_list = []
    drag_list = []

    for Umax in Umax_list:
        print(f"Re = {Umax/3*2*100:.1f}")
        Umax_const.assign(Umax)
        solver.solve() #update w
        solver_traction.solve() #update gg
        (drag_coords,drag_values,lift_coords,lift_values) = compute_traction_bndryval(GGspace,gg)

        if COMM_WORLD.rank == 0:
            dimless_const = 1.0
            drag_values = (np.array(drag_values) * (2.0/(0.2*0.2*0.1)) * dimless_const).tolist()
            lift_values = (np.array(lift_values) * (2.0/(0.2*0.2*0.1)) * dimless_const).tolist()

            # Save all into one compressed .npz file
            folder_name = f"reynolds_nb_data/Re={Umax/3*2*100:.1f}"
            print(f"Folder you want to access: {folder_name}")
            subdir_path = os.path.join(data_dir, folder_name)
            os.makedirs(subdir_path, exist_ok=True)
            traction_file = f"{subdir_path}/traction_data.npz"
            np.savez_compressed(
                traction_file,
                drag_coords=drag_coords,
                drag_values=drag_values,
                lift_coords=lift_coords,
                lift_values=lift_values
            )

        lift = assemble(-2.0/(0.2*0.2*0.1)*gg[1]*ds(5))
        print(f"{lift=}")
        drag = assemble(-2.0/(0.2*0.2*0.1)*gg[0]*ds(5))
        print(f"{drag=}")

        
        lift_list.append(lift)
        drag_list.append(drag)

    drag_lift_file = f"{data_dir}/Re_lift_drag_data.npz"
    np.savez_compressed(
        drag_lift_file,
        Re=np.array(Re_list),
        lift=np.array(lift_list),
        drag=np.array(drag_list)
    )
