import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
import turek_traction_utils
import warnings
import numpy as np
import os

if not os.path.exists(f"EOCplots/"):
    os.makedirs(f"EOCplots/")
if not os.path.exists(f"EOCdata/"):
    os.makedirs(f"EOCdata/")

# Suppress specific FutureWarning messages
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

# Set fd.info threshold
info = PETSc.Sys.Print
fd.logging.set_log_level(fd.DEBUG)  # fd.DEBUG, INFO, WARNING, ERROR, CRITICAL


################## Navier-Stokes - Turek Schafer Benchmark ##########################
# S or NS
stokes = False
stokes = os.getenv("STOKES", str(stokes)) == "True"

# Setup hierarchy of meshes
circle_init = 8  # init. nb. of points on circle bndry
max_ref = 5  # depth of mesh hierarchy
max_ref = int(os.getenv("MAX_REF", str(max_ref)))
order = 2  # geometry approximation order
order = int(os.getenv("ORDER", str(order)))

_, circle_points = turek_traction_utils.generate_ngmesh_manual(circle_init)
ngmsh = turek_traction_utils.generate_ngmesh_spline()
assert circle_init % 8 == 0, "circle_init must be divisible by 8"
circle_points = circle_points[::circle_init // 8]
subngmsh = turek_traction_utils.create_boundary_layer_submesh(ngmsh)
# fd.info(f"{circle_points=}")  # In these points we output pointwise traction values

mesh = fd.Mesh(ngmsh, comm=fd.COMM_WORLD)
if order > 1:
    cf = mesh.curve_field(order)
    mesh = fd.Mesh(cf)
submesh = fd.Mesh(subngmsh)
hierarchy, interpolation_hierarchy, subhierarchy = \
    turek_traction_utils.turek_computational_interpolation_hierarchy(ngmsh, max_ref, order)
hierarchy.insert(0, mesh)

# Create function spaces on hierarchies beforehand
interpolation_hierarchy_space_CG1 = []
interpolation_hierarchy_space_DG1 = []
for mesh in interpolation_hierarchy:
    interpolation_hierarchy_space_CG1.append(
        fd.VectorFunctionSpace(mesh, "CG", 1))
    interpolation_hierarchy_space_DG1.append(
        fd.VectorFunctionSpace(mesh, "DG", 1))

fd.info("Hierarchy created")

fd.info("------------------------------------------")
if stokes:
    fd.info("Computing EOC for stationary Stokes system")
else:
    fd.info("Computing EOC for stationary N-S equations")
fd.info("------------------------------------------")

# Build equations outside problem setup - we need them for PS computation
nu = fd.Constant(0.001)


def a(v, u, nu, stokes=False):
    Dv = 0.5*(fd.grad(v)+fd.grad(v).T)
    if stokes:
        return (fd.inner(2*nu*Dv, fd.grad(u)))*fd.dx
    else:
        return (fd.inner(fd.grad(v)*v, u) + fd.inner(2*nu*Dv, fd.grad(u)))*fd.dx


def b(q, v):
    return fd.inner(fd.div(v), q)*fd.dx


def F1(v, p, v_):
    return a(v, v_, nu, stokes) - b(p, v_)


def setup_problem(mesh, ref_level):
    # Build finite elements
    Ev = fd.VectorFunctionSpace(mesh, "CG", 2)  # velocity
    Ep = fd.FunctionSpace(mesh, "CG", 1)  # pressure

    # Build function spaces
    W = fd.MixedFunctionSpace([Ev, Ep])
    fd.info(f"Solving on level {ref_level+1} of {max_ref+1}")
    fd.info(f"#DoF = {W.dim()}")
    W_dim_list.append(W.dim())

    # Mesh quantities
    x = fd.SpatialCoordinate(mesh)
    fd.dx = fd.dx(degree=4)
    n = fd.FacetNormal(mesh)
    h = fd.CellDiameter(mesh)

    # Boundary conditions
    v_ref = fd.as_vector([4.0*0.3*x[1]*(0.41-x[1])/0.41/0.41, 0])
    noslip = fd.Constant((0.0, 0.0))
    freeslip = fd.Constant(0.0)
    bc_in = fd.DirichletBC(W.sub(0), v_ref, 1)
    bc_walls = fd.DirichletBC(W.sub(0), noslip, 2)
    bc_cylinder = fd.DirichletBC(W.sub(0), noslip, 5)
    bc_out = fd.DirichletBC(W.sub(0).sub(1), freeslip, 3)
    bcs = [bc_in, bc_walls, bc_cylinder, bc_out]

    # Build functions
    v_, p_ = fd.TestFunctions(W)
    w = fd.Function(W)
    v, p = fd.split(w)

    Eq = F1(v, p, v_) - b(p_, v)
    J = fd.derivative(Eq, w)

    # Solve using Newton and direct solver
    problem = fd.NonlinearVariationalProblem(Eq, w, bcs=bcs, J=J)
    lu = {"mat_type": "aij",
          "snes_type": "newtonls",
          "snes_monitor": None,
          "snes_converged_reason": None,
          "snes_max_it": 12,
          "snes_rtol": 1e-11,
          "snes_atol": 5e-10,
          "snes_linesearch_type": "basic",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=lu)

    if stokes:
        fd.info("Solving Stokes system")
    else:
        fd.info("Solving N-S equation")
    solver.solve()
    fd.info("Solved")

    # Initialize the files for computing point-wise traction
    v, p = w.subfunctions

    divv = fd.assemble(fd.div(v)**2*fd.dx)**0.5
    fd.info(f"{divv=}")

    return v, p


################## disrete variational traction ##########################
def compute_traction(submesh, mesh, v, p, ref_level):
    # Build spaces for traction (linear problem)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)  # Mind the degree
    g = fd.TrialFunction(V)
    g_ = fd.TestFunction(V)
    fd.info(f"{V.dim()=}")

    # set BC
    bc_list = [1,2,3]  # zero traction on bndries not touching bndry of interest
    bc_num = 5  # circle bndry - here we solve PS
    bcs_g = [fd.DirichletBC(V, fd.Constant((0, 0)), i) for i in bc_list]

    # Define the bilinear form for disrete variational traction problem
    a_g = fd.inner(g, g_) * fd.ds(bc_num)
    L_g = F1(v,
             p, g_)
    
    # Solve linear system
    gg = fd.Function(V)
    A = fd.assemble(a_g, bcs=bcs_g)
    b = fd.assemble(L_g)

    # Dirty ident_zeros trick
    diagonal = A.petscmat.getDiagonal()
    vals = diagonal.array
    vals[vals == 0] = 1
    A.petscmat.setDiagonal(diagonal, PETSc.InsertMode.INSERT_VALUES)

    fd.info("Solving disrete variational traction problem using sparse LU")
    fd.solve(A, gg, b, solver_parameters={"ksp_type": "preonly",
                                          "pc_type": "lu",
                                          "pc_factor_mat_solver_type": "mumps"})
    fd.info("PS solved")

    fd.info("Reference values for N-S from Turek Benchmark")
    drag_ref = 5.57953523384
    lift_ref = 0.010618948146
    fd.info(f"{drag_ref=}, {lift_ref=}")

    drag = fd.assemble(-2.0/(0.2*0.2*0.1)*gg[0]*fd.ds(bc_num))
    lift = fd.assemble(-2.0/(0.2*0.2*0.1)*gg[1]*fd.ds(bc_num))
    fd.info(f"direct computation: {drag=}, {lift=}")
    L1norm_error["t_h^dvt"]["drag_list"].append(drag)
    L1norm_error["t_h^dvt"]["lift_list"].append(lift)

    # Get pointwise traction values at points (W,NW,N,NE,E,SE,S,SW)
    mesh.tolerance = 0.1
    actual_refinement_but_same_points = []
    for point in circle_points:
        actual_refinement_but_same_points.append(
            (-2.0/(0.2*0.2*0.1))*gg.at(point))
    gg_points.append(actual_refinement_but_same_points)

############### Traction computation - direct ###################
    # List all the combinations for which convergence curved should be plotted
    I = fd.Identity(mesh.geometric_dimension())
    def force(v, p, normal): return (nu*(fd.grad(v)+fd.grad(v).T)-p*I)*normal

    # Analytical Unit outward normal cutted off to be local around cylinder
    x = fd.SpatialCoordinate(mesh)
    n = fd.as_vector([-(x[0]-0.2), -(x[1]-0.2)]) / \
        ((x[0]-0.2)**2+(x[1]-0.2)**2)**0.5  # 0.05

    # We compute dimensionless drag and lift (2 / U**2 / L * drag)
    g_tnds_DG1_analN = fd.assemble(interpolate(
        force(v, p, n), fd.VectorFunctionSpace(mesh, "DG", 1)))
    g_tnds_DG1_analN.rename("traction_interpolated_CG1_analN")
    dragTndS_analN = fd.assemble(-2.0/(0.2*0.2*0.1)
                                 * force(v, p, n)[0]*fd.ds(bc_num))
    liftTndS_analN = fd.assemble(-2.0/(0.2*0.2*0.1)
                                 * force(v, p, n)[1]*fd.ds(bc_num))
    L1norm_error["t(n_Omega)"]["drag_list"].append(
        dragTndS_analN)
    L1norm_error["t(n_Omega)"]["lift_list"].append(
        liftTndS_analN)

    # Projected FacetNormal on its natural space
    nn = fd.FacetNormal(mesh)
    n_facet_error = fd.sqrt(fd.assemble(fd.inner(n-nn, n-nn)*fd.ds(5)))
    fd.info(f"{n_facet_error=}")
    n_DG0 = turek_traction_utils.my_L2projection_bndry(
        nn, fd.VectorFunctionSpace(mesh, "DG", 0))
    n_DG0_error = fd.sqrt(fd.assemble(fd.inner(n-n_DG0, n-n_DG0)*fd.ds(5)))
    fd.info(f"{n_DG0_error=}")

    g_tnds_DG1_DG0n = fd.assemble(interpolate(
        force(v, p, n_DG0), fd.VectorFunctionSpace(mesh, "DG", 1)))
    g_tnds_DG1_DG0n.rename("traction_interpolated_CG1_DG0n")
    dragTndS_DG0n = fd.assemble(-2.0/(0.2*0.2*0.1)
                                * force(v, p, n_DG0)[0]*fd.ds(bc_num))
    liftTndS_DG0n = fd.assemble(-2.0/(0.2*0.2*0.1)
                                * force(v, p, n_DG0)[1]*fd.ds(bc_num))
    L1norm_error["t(n_Omega_h)"]["drag_list"].append(
        dragTndS_DG0n)
    L1norm_error["t(n_Omega_h)"]["lift_list"].append(
        liftTndS_DG0n)

    # Projected FacetNormal on smooth space
    nn = fd.FacetNormal(mesh)
    n_CG1 = turek_traction_utils.my_L2projection_bndry(
        nn, fd.VectorFunctionSpace(mesh, "CG", 1))
    n_CG1_error = fd.sqrt(fd.assemble(fd.inner(n-n_CG1, n-n_CG1)*fd.ds(5)))
    fd.info(f"{n_CG1_error=}")

    g_tnds_DG1_CG1n = fd.assemble(interpolate(
        force(v, p, n_CG1), fd.VectorFunctionSpace(mesh, "DG", 1)))
    g_tnds_DG1_CG1n.rename("traction_interpolated_CG1_CG1n")
    dragTndS_CG1n = fd.assemble(-2.0/(0.2*0.2*0.1)
                                * force(v, p, n_CG1)[0]*fd.ds(bc_num))
    liftTndS_CG1n = fd.assemble(-2.0/(0.2*0.2*0.1)
                                * force(v, p, n_CG1)[1]*fd.ds(bc_num))
    L1norm_error["t(P1 n_Omega_h)"]["drag_list"].append(
        dragTndS_CG1n)
    L1norm_error["t(P1 n_Omega_h)"]["lift_list"].append(
        liftTndS_CG1n)

    g_tnds_CG1 = turek_traction_utils.my_L2projection_bndry(
        force(v, p, n), fd.VectorFunctionSpace(mesh, "CG", 1))
    g_tnds_CG1.rename("traction_projected_CG1_analn")
    dragTndSCG1_analn = fd.assemble(-2.0 /
                                    (0.2*0.2*0.1)*g_tnds_CG1[0]*fd.ds(bc_num))
    liftTndSCG1_analn = fd.assemble(-2.0 /
                                    (0.2*0.2*0.1)*g_tnds_CG1[1]*fd.ds(bc_num))
    L1norm_error["P_1 t(n_Omega)"]["drag_list"].append(
        dragTndSCG1_analn)
    L1norm_error["P_1 t(n_Omega)"]["lift_list"].append(
        liftTndSCG1_analn)

    g_tnds_CG1_CG1n = turek_traction_utils.my_L2projection_bndry(
        force(v, p, n_CG1), fd.VectorFunctionSpace(mesh, "CG", 1))
    g_tnds_CG1_CG1n.rename("traction_projected_CG1_CG1n")
    dragTndSCG1_CG1n = fd.assemble(-2.0/(0.2*0.2*0.1)
                                   * g_tnds_CG1_CG1n[0]*fd.ds(bc_num))
    liftTndSCG1_CG1n = fd.assemble(-2.0/(0.2*0.2*0.1)
                                   * g_tnds_CG1_CG1n[1]*fd.ds(bc_num))
    L1norm_error["P_1 t(P_1 n_Omega_h)"]["drag_list"].append(
        dragTndSCG1_CG1n)
    L1norm_error["P_1 t(P_1 n_Omega_h)"]["lift_list"].append(
        liftTndSCG1_CG1n)

    return (gg, g_tnds_CG1, g_tnds_CG1_CG1n, g_tnds_DG1_analN, g_tnds_DG1_CG1n, g_tnds_DG1_DG0n)


# Lists to store L2 norms and dimensions
W_dim_list = []
L2norm_error = {
    "t_h^dvt": {
        "traction_fun_list": [],
        "error_list": []
    },
    "P_1 t(n_Omega)": {
        "traction_fun_list": [],
        "error_list": []
    },
    "P_1 t(P_1 n_Omega_h)": {
        "traction_fun_list": [],
        "error_list": []
    },
    "t(n_Omega)": {
        "traction_fun_list": [],
        "error_list": []
    },
    "t(P1 n_Omega_h)": {
        "traction_fun_list": [],
        "error_list": []
    },
    "t(n_Omega_h)": {
        "traction_fun_list": [],
        "error_list": []
    }
}

L1norm_error = {
    "t_h^dvt": {
        "drag_list": [],
        "lift_list": []
    },
    "P_1 t(n_Omega)": {
        "drag_list": [],
        "lift_list": []
    },
    "P_1 t(P_1 n_Omega_h)": {
        "drag_list": [],
        "lift_list": []
    },
    "t(n_Omega)": {
        "drag_list": [],
        "lift_list": []
    },
    "t(P1 n_Omega_h)": {
        "drag_list": [],
        "lift_list": []
    },
    "t(n_Omega_h)": {
        "drag_list": [],
        "lift_list": []
    }
}

labels_list = list(L2norm_error.keys())

gg_points = []

ref_level = 0
for mesh, submesh in zip(hierarchy, subhierarchy):
    fd.info("-------------------------------------------------")
    fd.info("Loading mesh from hierarchy")
    v, p = setup_problem(mesh, ref_level)
    (gg_fun, g_tnds_CG1_fun, g_tnds_CG1_inter_fun, g_tnds_DG1_analN_fun, g_tnds_DG1_CG1n_fun,
     g_tnds_DG1_DG0n_fun) = compute_traction(submesh, mesh, v, p, ref_level)

    # Append functions to the corresponding lists in the dictionary
    L2norm_error["t_h^dvt"]["traction_fun_list"].append(
        turek_traction_utils.visprolong(gg_fun))
    L2norm_error["P_1 t(n_Omega)"]["traction_fun_list"].append(
        turek_traction_utils.visprolong(g_tnds_CG1_fun))
    L2norm_error["P_1 t(P_1 n_Omega_h)"]["traction_fun_list"].append(
        turek_traction_utils.visprolong(g_tnds_CG1_inter_fun))
    L2norm_error["t(n_Omega)"]["traction_fun_list"].append(
        turek_traction_utils.visprolong(g_tnds_DG1_analN_fun))
    L2norm_error["t(P1 n_Omega_h)"]["traction_fun_list"].append(
        turek_traction_utils.visprolong(g_tnds_DG1_CG1n_fun))
    L2norm_error["t(n_Omega_h)"]["traction_fun_list"].append(
        turek_traction_utils.visprolong(g_tnds_DG1_DG0n_fun))

    fd.info("Tractions computed and saved to list")
    ref_level += 1

############### Now we compute correct || t - t_h ||_L2, where t is refernce on finest refinement ###############


def interpolate_to_get_L2norm_error_iterative(traction_fun_list, error_list, list_name, t_ref_list):
    """ Idea: We must compare values of the coarse-mesh solution on the boundary
        with the boundary values of the fine-grid reference solution. This is not
        default when using straighforward interpolataion or projection between
        nested mesh hierarchy with a curved boundary. When we refine mesh and move
        vertices to correct the boundary, then such mesh is fully contained in the
        coarse mesh (since our mesh is non-convex where the curved bndry is). Hence,
        if we interpolate from coarse to fine mesh, we pass to the fine mesh values
        from the interior of the cells. The right approach is to take the coarse
        solution, refine its mesh, interpolate function and then move vertices.
        This is done iteratively if there is more than one refinement level
        between meshes. We do it as cheap as possible since in the interpolation_hierarchy
        there are meshes only with boundary and adjanced elements. You cant do better.
    """
    t_ref = t_ref_list[-1]
    traction_fun_list_last = traction_fun_list[-1]
    traction_fun_list = traction_fun_list[:-1]
    bc_num = 5
    max_len = len(traction_fun_list)
    for i in range(0, max_len):
        t_h = traction_fun_list[i]
        for k in range(i, max_len):
            refined_mesh = fd.Mesh(
                interpolation_hierarchy[k].coordinates.copy(deepcopy=True))  # crucial

            # Sort auxiliary interpolation by their correct space to be error-free
            if list_name.startswith("t_h") or list_name.startswith("P_1 t"):
                V_new = interpolation_hierarchy_space_CG1[k]
            elif list_name.startswith("t("):
                V_new = interpolation_hierarchy_space_DG1[k]
            else:
                raise ValueError(f"Unexpected list name: {list_name}")

            t_h_refined = fd.assemble(interpolate(t_h, V_new))

            turek_traction_utils.move_boundary_to_circle_firedrake(refined_mesh)
            t_h_refined_moved = fd.Function(fd.functionspaceimpl.WithGeometry.create(
                V_new, refined_mesh), val=t_h_refined.topological)

            t_h = t_h_refined_moved

        # This should be no-op: transfer t_ref to the boundary-adjacent cells
        # mesh. To be sure, we could keep t_ref and instead set the integration
        # domain properly.
        _t_ref = interpolate(t_ref, t_h.function_space())

        fd.info("Traction interpolated")
        L2normError = fd.assemble(fd.inner(t_h-_t_ref, t_h-_t_ref)*fd.ds(bc_num)) ** 0.5
        fd.info(f"{list_name}={L2normError}")
        error_list.append(L2normError)


# Before doing the most expensive job, evaluate drag and lift errors
fd.info("---------------------------------------- \n Saving drag and lift")
plot_type = "S" if stokes else "NS"
if stokes:
    W_dim_list_shorter = W_dim_list[:-1]
    drag_ref = L1norm_error["t_h^dvt"]["drag_list"][-1]
    lift_ref = L1norm_error["t_h^dvt"]["lift_list"][-1]
    dragerror_compilation = [
        [abs(val - drag_ref) for val in L1norm_error[key]["drag_list"][:-1]]
        for key in labels_list
    ]  # Subtract last element of "t_h^dvt" from each other
    lifterror_compilation = [
        [abs(val - lift_ref) for val in L1norm_error[key]["lift_list"][:-1]]
        for key in labels_list
    ]  # Subtract last element of "t_h^dvt" from each other
    turek_traction_utils.traction_EOC_plot(
        W_dim_list_shorter,
        dragerror_compilation,
        f"EOCplots/convergence_plot_drag_{plot_type}",
        labels_list
    )
    turek_traction_utils.traction_EOC_plot(
        W_dim_list_shorter,
        lifterror_compilation,
        f"EOCplots/convergence_plot_lift_{plot_type}",
        labels_list
    )
    # Save results to CSV
    turek_traction_utils.save_csv(
        W_dim_list_shorter,
        dragerror_compilation,
        labels_list,
        filename=f"EOCdata/convergence_data_dragerror_{plot_type}.csv"
    )
    turek_traction_utils.save_csv(
        W_dim_list_shorter,
        lifterror_compilation,
        labels_list,
        filename=f"EOCdata/convergence_data_lifterror_{plot_type}.csv"
    )
    drag_compilation = [L1norm_error[key]["drag_list"] for key in labels_list]
    lift_compilation = [L1norm_error[key]["lift_list"] for key in labels_list]
    turek_traction_utils.save_csv(
        W_dim_list,
        drag_compilation,
        labels_list,
        filename=f"EOCdata/convergence_data_drag_{plot_type}.csv"
    )
    turek_traction_utils.save_csv(
        W_dim_list,
        lift_compilation,
        labels_list,
        filename=f"EOCdata/convergence_data_lift_{plot_type}.csv"
    )
else:
    drag_ref = 5.57953523384
    lift_ref = 0.010618948146
    dragerror_compilation = [
        [abs(val - drag_ref) for val in L1norm_error[key]["drag_list"]]
        for key in labels_list
    ]  # Subtract reference value but keep all elements
    lifterror_compilation = [
        [abs(val - lift_ref) for val in L1norm_error[key]["lift_list"]]
        for key in labels_list
    ]  # Subtract reference value but keep all elements
    turek_traction_utils.traction_EOC_plot(
        W_dim_list,
        dragerror_compilation,
        f"EOCplots/convergence_plot_drag_{plot_type}",
        labels_list
    )
    turek_traction_utils.traction_EOC_plot(
        W_dim_list,
        lifterror_compilation,
        f"EOCplots/convergence_plot_lift_{plot_type}",
        labels_list
    )
    # Save results to CSV
    turek_traction_utils.save_csv(
        W_dim_list,
        dragerror_compilation,
        labels_list,
        filename=f"EOCdata/convergence_data_dragerror_{plot_type}.csv"
    )
    turek_traction_utils.save_csv(
        W_dim_list,
        lifterror_compilation,
        labels_list,
        filename=f"EOCdata/convergence_data_lifterror_{plot_type}.csv"
    )
    drag_compilation = [L1norm_error[key]["drag_list"] for key in labels_list]
    lift_compilation = [L1norm_error[key]["lift_list"] for key in labels_list]
    turek_traction_utils.save_csv(
        W_dim_list,
        drag_compilation,
        labels_list,
        filename=f"EOCdata/convergence_data_drag_{plot_type}.csv"
    )
    turek_traction_utils.save_csv(
        W_dim_list,
        lift_compilation,
        labels_list,
        filename=f"EOCdata/convergence_data_lift_{plot_type}.csv"
    )
fd.info("Saving drag and lift --- Done \n ----------------------------------------")

fd.info("---------------------------------------- \n Saving pointwise drag and lift")

selected_indices = [0, 2, 4, 6]
num_refinements = len(gg_points)
num_selected = len(selected_indices)

# Initialize arrays to store drag and lift values
stacked_drag = np.zeros((num_refinements, num_selected))
stacked_lift = np.zeros((num_refinements, num_selected))

# Extract values
for i, refinement_data in enumerate(gg_points):  # Loop over refinements
    for j, point_idx in enumerate(selected_indices):  # Loop over selected points
        # Get drag/lift at the selected point
        pointwise_drag, pointwise_lift = refinement_data[point_idx]
        stacked_drag[i, j] = pointwise_drag
        stacked_lift[i, j] = pointwise_lift

data_lists = (
    [L1norm_error["t_h^dvt"]["drag_list"],
     L1norm_error["t_h^dvt"]["lift_list"]]
    + stacked_drag.T.tolist()
    + stacked_lift.T.tolist()
)

latex_values_table_labels = ["drag", "lift"] + [f"Pointwise drag {idx+1}" for idx in selected_indices] + [
    f"Point lift {idx+1}" for idx in selected_indices]
turek_traction_utils.generate_latex_table(W_dim_list, latex_values_table_labels,
                            data_lists, f"EOCdata/pointwise_values_{plot_type}")

fd.info("---------------------------------------- \n Getting L2 pointwise traction error")

# Iterate through the dictionary and process each dataset
for name, data in L2norm_error.items():
    fd.info("------------------------------------------")
    interpolate_to_get_L2norm_error_iterative(
        data["traction_fun_list"],
        data["error_list"],
        name,
        L2norm_error["t_h^dvt"]["traction_fun_list"]
    )

W_dim_list = W_dim_list[:-1]
L2norms_compilation = [L2norm_error[key]["error_list"]
                       for key in labels_list]  # Extract error lists

# Save results to CSV
turek_traction_utils.save_csv(
    W_dim_list,
    L2norms_compilation,
    labels_list,
    filename=f"EOCdata/convergence_data_{plot_type}.csv"
)

# Generate EOC plot
turek_traction_utils.traction_EOC_plot(
    W_dim_list,
    L2norms_compilation,
    f"EOCplots/convergence_plot_{plot_type}",
    labels_list
)

fd.info("Konec / End / Ende")
