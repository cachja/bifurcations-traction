import firedrake as fd
from firedrake.petsc import PETSc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def my_L2projection_bndry(n, V):
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    a = fd.inner(u, v)*fd.ds
    l = fd.inner(n, v)*fd.ds
    A = fd.assemble(a)
    L = fd.assemble(l)

    diagonal = A.petscmat.getDiagonal()
    vals = diagonal.array
    vals[vals == 0] = 1
    A.petscmat.setDiagonal(diagonal, PETSc.InsertMode.INSERT_VALUES)

    nh = fd.Function(V)

    fd.solve(A, nh.vector(), L)
    return nh


def move_boundary_to_circle_firedrake(mesh, boundary_marker=5, center=(0.2, 0.2), radius=0.05):
    """Move the vertices corresponding to boundary_marker to a circle (parallel-safe)."""

    # Access mesh coordinates (including halos for safe parallel access)
    coordinates = mesh.coordinates.dat.data_with_halos[:]

    # Create a function space for identifying boundary nodes
    V = fd.FunctionSpace(mesh, "CG", 1)
    bc = fd.DirichletBC(V, fd.Constant(1.0), boundary_marker)

    # Get boundary nodes (global indices)
    boundary_nodes = bc.nodes

    # Get process-local indices (avoid out-of-bounds errors in parallel)
    local_size = len(mesh.coordinates.dat.data)  # Number of local dofs
    local_boundary_nodes = [v for v in boundary_nodes if v < local_size]

    # Move the boundary nodes
    for vertex in local_boundary_nodes:
        # Get vertex coordinates safely
        x, y = coordinates[vertex]

        # Compute displacement to project onto the circle
        dx = x - center[0]
        dy = y - center[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Scale the point to the circle radius
        new_x = center[0] + (radius * dx / distance)
        new_y = center[1] + (radius * dy / distance)

        # Update only the local part of the data
        mesh.coordinates.dat.data[vertex] = [new_x, new_y]

    # Firedrake automatically synchronizes mesh coordinates across processes
    return fd.Mesh(mesh.coordinates)


def move_boundary_to_circle_netgen(ngmsh, bndry_num, center=(0.2, 0.2), radius=0.05):
    import numpy as np
    from netgen.meshing import MeshPoint
    for el in ngmsh.Elements1D():
        # get boundary index
        if el.index == bndry_num:
            # get name of boundary condition (one based)
            # print(ngmsh.GetBCName(el.index-1))
            # get vertex numbers of element
            point_index = el.vertices[0]
            # get vertex coordinates
            coordinates = ngmsh[el.vertices[0]]

            # Get vertex coordinates
            x, y, z = coordinates

            # Compute new coordinates to move the point back to the circle
            dx = x - center[0]
            dy = y - center[1]
            distance = np.sqrt(dx**2 + dy**2)

            # Scale the point to the circle radius
            new_x = center[0] + (radius * dx / distance)
            new_y = center[1] + (radius * dy / distance)

            # Move the vertex to the new coordinates
            ngmsh.Points()[point_index] = MeshPoint((new_x, new_y, z))


def get_EOC(list_DOFnum, list_drags):
    log_x = np.log(np.sqrt(list_DOFnum))
    log_y = np.log(list_drags)
    slope, intercept, tmp1, tmp2, tmp3 = linregress(log_x, log_y)
    fd.info(f"{-slope=}")
    return (slope, intercept)


def traction_EOC_plot(DoF, L2norm, name, labels, title=None):
    import numpy as np
    import matplotlib.pyplot as plt

    markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+', 'p', 'h']
    sqrt_DoF = np.sqrt(DoF)
    plt.xlabel(r'$\sqrt{DoF}$')
    plt.ylabel(r'$|| t - t_{manufactured} ||_{L^2(\partial\Omega)}$')
    plt.xscale('log')
    plt.yscale('log')
    if title != None:
        plt.title(title)

    partial_slopes = []  # To store the partial EOC fits for each dataset

    def get_partial_EOC(DoF, drags):
        """
        Computes partial EOC fits (slope and intercept) for two consecutive points.
        Returns a list of tuples: [(slope1, intercept1), (slope2, intercept2), ...].
        """
        partial_fits = []
        log_x = np.log(np.sqrt(DoF))  # sqrt(DoF) in log space
        log_y = np.log(drags)

        for i in range(len(DoF) - 1):
            # Fit only two points
            slope, intercept, _, _, _ = linregress(log_x[i:i+2], log_y[i:i+2])
            partial_fits.append(slope)

        return partial_fits

    for i in range(len(L2norm)):
        fd.info(f"plotting {labels[i]}")
        partial_slope_list = get_partial_EOC(DoF, L2norm[i])

        # Store slopes for the current label
        partial_slopes.append(partial_slope_list)

        # Plot the data
        plt.plot(
            sqrt_DoF, L2norm[i], label=f'{labels[i]}, last EOC = {round(-partial_slope_list[-1], 2)}', marker=markers[i])

    plt.legend()
    plt.savefig(f'{name}.pdf', bbox_inches='tight', dpi=300)
    plt.clf()

    # Generate LaTeX table
    latex_table = []
    latex_table.append(r"\begin{table}[h!]")
    latex_table.append(r"\centering")
    latex_table.append(r"\begin{tabular}{|c|" + "c|" * len(labels) + r"}")
    latex_table.append(r"\hline")
    header = "DoF & " + " & ".join(labels) + r" \\ \hline"
    latex_table.append(header)

    # Transpose `partial_slopes` to align rows by refinement level
    for i, dof in enumerate(DoF[1:]):  # Skip the first DoF
        row = f"{dof} & " + " & ".join(
            f"{-partial_slopes[j][i]:.2f}" for j in range(len(labels))) + r" \\ \hline"
        latex_table.append(row)

    latex_table.append(r"\end{tabular}")
    latex_table.append(r"\caption{Convergence results.}")
    latex_table.append(r"\label{tab:fem_convergence}")
    latex_table.append(r"\end{table}")

    # Print the LaTeX table
    fd.info("\n".join(latex_table))
    with open(name + "_latex.txt", 'w') as f:
        f.write("\n".join(latex_table))  # Save to file

    return partial_slopes


def save_csv(W_dim_list, L2norms_compilation, labels, filename="convergence_data.csv"):
    import csv
    # Open a CSV file for writing
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the headers: "DoF" and the labels
        writer.writerow(["DoF"] + labels)

        # Write the data rows, where each row corresponds to a DoF and its respective L2 norms
        for i in range(len(W_dim_list)):
            row = [W_dim_list[i]]  # Start with the DoF value
            for l2norms in L2norms_compilation:
                # Add the corresponding L2norm for this DoF
                row.append(l2norms[i])
            writer.writerow(row)


def generate_latex_table(dofs, labels, data_lists, output_filename):
    """
    Generate a LaTeX table and save it to a file.

    Parameters:
    - dofs: list of degrees of freedom (first column)
    - labels: list of column labels (excluding "DoF")
    - data_lists: list of lists, each corresponding to a column in the table
    - output_filename: name of the output file (without extension)
    """

    # Validate input sizes
    assert len(labels) == len(
        data_lists), "Number of labels must match number of data lists."
    assert all(len(data) == len(dofs)
               for data in data_lists), "Each data list must have the same length as dofs."

    # Initialize LaTeX table structure
    latex_table = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\begin{tabular}{|c|" + "c|" * len(labels) + r"}",
        r"\hline"
    ]

    # Create header row
    header = "DoF & " + " & ".join(labels) + r" \\ \hline"
    latex_table.append(header)

    # Add data rows
    for i, dof in enumerate(dofs):
        row_values = [f"{data_lists[j][i]:.8f}" for j in range(len(labels))]
        row = f"{dof} & " + " & ".join(row_values) + r" \\ \hline"
        latex_table.append(row)

    # Finalize table
    latex_table.extend([
        r"\end{tabular}",
        r"\caption{Convergence results.}",
        r"\label{tab:fem_convergence}",
        r"\end{table}"
    ])

    # Convert list to string for output
    latex_string = "\n".join(latex_table)

    # Print table to console
    fd.info(latex_string)

    # Save to file
    with open(f"{output_filename}_latex.txt", "w") as f:
        f.write(latex_string)


def visprolong(u):
    return u.copy(deepcopy=True)


def generate_ngmesh_manual(num_points):
    from netgen.geom2d import SplineGeometry
    import numpy as np

    geo = SplineGeometry()

    # Add rectangle
    geo.AddRectangle((0, 0), (2.2, 0.41), bcs=(2, 3, 2, 1))

    # Define the circle with points
    # num_points = 8  # More points for a smoother circle
    radius = 0.05
    center = (0.2, 0.2)

    # Generate points for the circle
    circle_points = [
        (center[0] + radius * np.cos(2 * np.pi * i / num_points),
         center[1] + radius * np.sin(2 * np.pi * i / num_points))
        for i in range(num_points)
    ]
    # print(f"{circle_points=}")

    # Add points for the circle
    point_indices = []
    for point in circle_points:
        p = geo.AddPoint(point[0], point[1])
        point_indices.append(p)

    # Add line segments to form the circle
    for i in range(num_points):
        geo.Append(["line", point_indices[i], point_indices[(i + 1) % num_points]],
                   leftdomain=0, rightdomain=1, bc=5)  # Connect points in a loop

    # Generate the mesh
    mesh = geo.GenerateMesh(maxh=0.2)
    return mesh, circle_points


def create_boundary_layer_submesh(mother_mesh, boundary_index=5, new_boundary_index=6):
    from netgen.meshing import Mesh, MeshPoint, Element2D, Element1D, FaceDescriptor
    from netgen.csg import Pnt

    submesh = Mesh()
    submesh.dim = 2

    # Step 1: Find all vertex indices connected to the given boundary
    boundary_vertices = set()
    for bel in mother_mesh.Elements1D():
        if bel.index == boundary_index:
            boundary_vertices.update(bel.points)

    # Step 2: Find all 2D elements that share any of those vertices
    included_elements = []
    included_vertices = set()
    for el in mother_mesh.Elements2D():
        if any(p in boundary_vertices for p in el.points):
            included_elements.append(el)
            included_vertices.update(el.points)

    # Step 3: Map vertex indices to new mesh
    pnums = {}
    for vi in included_vertices:
        p = mother_mesh.Points()[vi]
        pnums[vi] = submesh.Add(MeshPoint(Pnt(p.p[0], p.p[1], p.p[2])))

    # Step 4: Add face descriptors
    # Original boundary
    submesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=boundary_index))
    # Optional new boundary
    submesh.Add(FaceDescriptor(surfnr=2, domin=1, bc=new_boundary_index))

    # Step 5: Add 2D elements
    for el in included_elements:
        new_pts = [pnums[pi] for pi in el.points]
        submesh.Add(Element2D(index=1, vertices=new_pts))

    # Step 6: Add boundary edges, remapping index if needed
    from collections import defaultdict

    # First, track all edges of included 2D elements
    edge_counter = defaultdict(list)

    for el in included_elements:
        pts = el.points
        edges = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])]
        for a, b in edges:
            # Use .nr to access the integer index
            edge = tuple(sorted((a.nr, b.nr)))
            edge_counter[edge].append(el)

    # Then detect boundary edges (edges only used once)
    for edge, attached_elements in edge_counter.items():
        if len(attached_elements) == 1:
            a_nr, b_nr = edge
            if a_nr in pnums and b_nr in pnums:
                v0, v1 = pnums[a_nr], pnums[b_nr]
                # Check if this edge existed in the original boundary
                original_bel = next(
                    (bel for bel in mother_mesh.Elements1D()
                     if set(p.nr for p in bel.points) == {a_nr, b_nr}),
                    None
                )
                bc = boundary_index if original_bel and original_bel.index == boundary_index else new_boundary_index
                submesh.Add(Element1D([v0, v1], index=bc))

    # Finalize mesh
    submesh.SetMaterial(1, "domain")

    return submesh


def turek_computational_interpolation_hierarchy_former(ngmsh, max_ref):
    ngmsh.SetGeometry(None)
    hierarchy = [fd.Mesh(ngmsh)]
    interpolation_hierarchy = []
    subhierarchy = [fd.Mesh(create_boundary_layer_submesh(ngmsh))]

    for i in range(max_ref):
        ngmsh.Refine()
        interpolation_hierarchy.append(
            fd.Mesh(create_boundary_layer_submesh(ngmsh)))
        move_boundary_to_circle_netgen(ngmsh, 5)
        hierarchy.append(fd.Mesh(ngmsh))
        subhierarchy.append(fd.Mesh(create_boundary_layer_submesh(ngmsh)))

    return (hierarchy, interpolation_hierarchy, subhierarchy)

def turek_computational_interpolation_hierarchy(ngmsh, max_ref):
    hierarchy = []
    interpolation_hierarchy = []
    subhierarchy = [fd.Mesh(create_boundary_layer_submesh(ngmsh))]

    for i in range(max_ref):
        ngmsh.Refine()
        interpolation_hierarchy.append(
            fd.Mesh(create_boundary_layer_submesh(ngmsh),comm=fd.COMM_WORLD))
        hierarchy.append(fd.Mesh(fd.Mesh(ngmsh,comm=fd.COMM_WORLD).curve_field(2)))
        subhierarchy.append(fd.Mesh(create_boundary_layer_submesh(ngmsh)))

    return (hierarchy, interpolation_hierarchy, subhierarchy)

def generate_ngmesh_spline():
    from netgen.geom2d import SplineGeometry
    import netgen
    if fd.COMM_WORLD.rank == 0:
        geo = SplineGeometry()
        geo.AddRectangle( (0, 0), (2.2, 0.41), bcs = (2, 3, 2, 1))
        geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc=5)#, maxh = 0.01
        ngmesh = geo.GenerateMesh(maxh=0.2)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(2)
    return ngmesh