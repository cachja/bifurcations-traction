from firedrake import *
import os
import re
import matplotlib.pyplot as plt
import numpy as np

comm = COMM_WORLD
rank = comm.Get_rank()


def plot_2d_contours_with_derivatives(coordinates, list_of_results, labels, name, path, label_var, hlines, Obs_y, problem, turning_points=None, smoothing=False):
    import numpy as np
    import matplotlib.pyplot as plt

    # Separate upper and lower coordinates and corresponding results
    x_upper, y_upper, sorted_results_upper, x_lower, y_lower, sorted_results_lower = sort_and_split_data(coordinates, list_of_results, Obs_y)

    # Compute curved distances
    curved_upper = np.array(calculate_cumulative_length(list(zip(x_upper, y_upper)), starting_point=(0.15,Obs_y)))
    curved_lower = -np.array(calculate_cumulative_length(list(zip(x_lower, y_lower)), starting_point=(0.15,Obs_y)))  # Flip to negative

    # Reynolds numbers
    Re_vals = np.array(labels, dtype=float)

    # Z-values
    Z_upper = np.array(sorted_results_upper)
    Z_lower = np.array(sorted_results_lower)

    # Merge along the curved distance axis
    curved_combined = np.concatenate((curved_lower[::-1], curved_upper))  # ensure correct order: left to right
    Z = np.concatenate((Z_lower[:, ::-1], Z_upper), axis=1)  # match curved_combined shape

    # Create meshgrid
    X, Y = np.meshgrid(curved_combined, Re_vals)
    X = X / 0.05

    # Derivatives
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    print(f"dx={dx}, dy={dy}")

    if smoothing:
        from scipy.signal import savgol_filter
        Z_smooth = savgol_filter(
            Z,
            window_length=11,   # must be odd
            polyorder=3,
            axis=1             # along boundary direction
        )
        dZ_dRe, dZ_dx = np.gradient(Z_smooth, dy, curved_combined, edge_order=2)
    else:
        dZ_dRe, dZ_dx = np.gradient(Z, dy, dx, edge_order=2)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        #"font.size": 20,
        # 'axes.titlesize': 14,   # Title font size
         'axes.labelsize': 18,   # X and Y labels font size
        # 'xtick.labelsize': 14,  # X-axis tick labels font size
        # 'ytick.labelsize': 14,  # Y-axis tick labels font size
        'legend.fontsize': 26,   # Legend font size
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    # Begin plotting
    #fig, ax = plt.subplots(figsize=(10, 16))
    fig, ax = plt.subplots(figsize=(5, 4))

    import matplotlib.lines as mlines
    def plot_zero_contour(Z, color, label, linestyle):
        ax.contour(X, Y, Z, levels=[0], colors=color, linestyles=[linestyle])
        # No return needed

    # Plot contours
    plot_zero_contour(dZ_dRe, 'black', fr'$\partial_{{Re}} {label_var} = 0$', "-")

    # Create proxy artists for legend
    legend_handles = [
        mlines.Line2D([], [], color='black', linestyle='-', label = fr'$\partial_{{Re}} {label_var} = 0$'),
    ]

    if turning_points:
        turning_points = np.array(turning_points)
        ax.scatter(turning_points[:, 1], turning_points[:, 0], color='red', marker='x', s=60, label='Turning Points')
        legend_handles.append(mlines.Line2D([], [], color='red', marker='x', linestyle='None', markersize=8, label='Turning Points'))

    ax.legend(handles=legend_handles, loc='upper right', fontsize=10, facecolor='white', framealpha=1)

    # Turning/extrema points
    if turning_points:
        turning_points = np.array(turning_points)
        ax.scatter(turning_points[:, 1], turning_points[:, 0], color='red', marker='x', s=60, label='Turning Points')

    ax.set_xlabel(r'$\theta$ [rad]')
    ax.set_ylabel('Reynolds number [1]')
    
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    yticks = np.arange(0, 1001, 100)
    # Combine and sort unique values with hlines
    hlines_re = np.array([p[1] for p in hlines], dtype=float)
    new_yticks = np.unique(np.concatenate((yticks, hlines_re)))

    # Set the new yticks
    ax.set_yticks(new_yticks)
    ax.grid(True, color='lightgray', linestyle=':', linewidth=1.0)

    ax.set_ylim(0,500)

    for (theta,h) in hlines:
        plt.axhline(y=h, color='red', linestyle=(0,(1,1)), linewidth=2.0)

    from matplotlib.lines import Line2D

    # Add short lines centered at turning points
    hlength = np.pi / 4
    vlength = 100

    for i, (theta, Re) in enumerate(hlines):
        if 6<Re < 50:  # First two: horizontal lines
            ax.add_line(Line2D([theta - hlength / 2, theta + hlength / 2], [Re, Re],
                            color='orange', linestyle='-', linewidth=1.5, alpha = 0.8, zorder = 2))
        else:  # Remaining: vertical lines
            ax.add_line(Line2D([theta, theta], [Re - vlength / 2, Re + vlength / 2],
                            color='orange', linestyle='-', linewidth=1.5, alpha = 0.8, zorder = 2))

    plt.tight_layout()
    #plt.title(name)
    plt.savefig(f"{path}_2D_{problem}.pdf")
    plt.close()
    print(f"✅ Saved 2D contour plot to {path}_2D_{problem}.pdf")
    """
    Merge drag coordinates and drag values from multiple MPI ranks.

    Parameters:
    - drag_coords_list: List of lists where each list contains drag coordinates from a rank.
    - drag_values_list_list: List of lists where each list contains drag values from a rank.

    Returns:
    - merged_drag_coords: Flattened list of drag coordinates from all ranks.
    - merged_drag_values_list: List of lists where each list contains concatenated drag values for each component.
    """

    # Gather all data from all ranks
    all_drag_coords = comm.allgather(drag_coords)
    all_drag_values_list = comm.allgather(drag_values_list)

    # Flatten the list of coordinates
    merged_drag_coords = [coord for rank_coords in all_drag_coords for coord in rank_coords]

    # Merging drag_values_list properly
    # First initialize an empty list for the merged drag values list
    if len(drag_values_list) > 0:
        num_components = len(drag_values_list)
        merged_drag_values_list = [[] for _ in range(num_components)]

        for rank_drag_values_list in all_drag_values_list:
            for i in range(num_components):
                merged_drag_values_list[i].extend(rank_drag_values_list[i])

    else:
        # If drag_values_list is empty for some reason, just handle it gracefully
        merged_drag_values_list = []

    return merged_drag_coords, merged_drag_values_list

def calculate_cumulative_length(points, starting_point):
    """
    Calculate the cumulative length of a curve given as a list of (x, y) points.

    Parameters:
    - points: List of tuples/lists [(x1, y1), (x2, y2), ...]
    - starting_point: Optional (x, y) to be prepended before points[0]

    Returns:
    - List of cumulative lengths, same length as `points`
    """
    def distance(p1, p2):
        return np.hypot(p2[0] - p1[0], p2[1] - p1[1])

    if starting_point is not None:
        points = [starting_point] + list(points)

    cumulative_lengths = [0.0]  # Always start at 0

    for i in range(1, len(points)):
        cumulative_lengths.append(cumulative_lengths[-1] + distance(points[i-1], points[i]))

    # Remove the prepended start length if we inserted starting_point
    return cumulative_lengths[1:] if starting_point is not None else cumulative_lengths

def sort_and_split_data_old(coordinates, list_of_results, Obs_y):
    # Separate upper and lower coordinates and results based on y value
    x_upper, y_upper = [], []
    x_lower, y_lower = [], []
    
    results_upper = [[] for _ in range(len(list_of_results))]
    results_lower = [[] for _ in range(len(list_of_results))]

    for (x, y), *results in zip(coordinates, *list_of_results):
        if y <= Obs_y:
            x_lower.append(x)
            y_lower.append(y)
            for i, result in enumerate(results):
                results_lower[i].append(result)
        else:  # Upper wing boundary
            x_upper.append(x)
            y_upper.append(y)
            for i, result in enumerate(results):
                results_upper[i].append(result)
    
    # Sorting upper and lower wing data by x-coordinate
    x_upper_unsorted = x_upper
    x_lower_unsorted = x_lower
    x_upper, y_upper = zip(*sorted(zip(x_upper, y_upper)))
    x_lower, y_lower = zip(*sorted(zip(x_lower, y_lower)))
    
    # Sort each list of results according to the sorted x-coordinates
    sorted_results_upper = []
    sorted_results_lower = []
    
    for results in results_upper:
        sorted_results_upper.append([result for x, result in sorted(zip(x_upper_unsorted, results))])
    
    for results in results_lower:
        sorted_results_lower.append([result for x, result in sorted(zip(x_lower_unsorted, results))])

    
    return x_upper, y_upper, sorted_results_upper, x_lower, y_lower, sorted_results_lower


def sort_and_split_data(coordinates, list_of_results, Obs_y):

    # ----------------------------
    # 1. Split into upper / lower
    # ----------------------------
    x_upper, y_upper = [], []
    x_lower, y_lower = [], []

    results_upper = [[] for _ in range(len(list_of_results))]
    results_lower = [[] for _ in range(len(list_of_results))]

    for (x, y), *results in zip(coordinates, *list_of_results):
        if y <= Obs_y:
            x_lower.append(x)
            y_lower.append(y)
            for i, result in enumerate(results):
                results_lower[i].append(result)
        else:
            x_upper.append(x)
            y_upper.append(y)
            for i, result in enumerate(results):
                results_upper[i].append(result)

    # ----------------------------
    # 2. Greedy ordering function
    # ----------------------------
    def order_with_values(points, values_list):
        points = np.array(points)
        values_list = [np.array(v) for v in values_list]

        n = len(points)

        # ✅ start from leftmost point
        start_idx = np.argmin(points[:, 0])

        order = [start_idx]
        used = set(order)
        current_idx = start_idx

        for _ in range(n - 1):
            current_point = points[current_idx]

            # distances to all points
            dists = np.linalg.norm(points - current_point, axis=1)

            # mask used points
            for idx in used:
                dists[idx] = np.inf

            # pick nearest unused
            next_idx = np.argmin(dists)

            order.append(next_idx)
            used.add(next_idx)
            current_idx = next_idx

        ordered_points = points[order]
        ordered_values = [v[order] for v in values_list]

        return ordered_points, ordered_values


    # ----------------------------
    # 3. Apply ordering
    # ----------------------------
    upper_points = list(zip(x_upper, y_upper))
    lower_points = list(zip(x_lower, y_lower))

    if len(upper_points) > 0:
        upper_ordered, sorted_results_upper = order_with_values(
            upper_points, results_upper
        )
        x_upper, y_upper = upper_ordered[:, 0], upper_ordered[:, 1]
    else:
        sorted_results_upper = results_upper

    if len(lower_points) > 0:
        lower_ordered, sorted_results_lower = order_with_values(
            lower_points, results_lower
        )
        x_lower, y_lower = lower_ordered[:, 0], lower_ordered[:, 1]
    else:
        sorted_results_lower = results_lower

    return x_upper, y_upper, sorted_results_upper, x_lower, y_lower, sorted_results_lower


if __name__ == "__main__":

    Wid = 0.41
    Rad = 0.05
    problem_list = ["cylinder","cylinder_symm","flower","flower_symm","cylinder_symm_wide","cylinder_wide"]
    problem = problem_list[4]
    if problem == "cylinder":
        data_dir = "./output_turek_cylinder/"
        Var = 0.0
        Obs_y = 0.2

    elif problem == "cylinder_symm":
        data_dir = "./output_turek_cylinder_symm/"
        Var = 0.0
        Obs_y = Wid/2

    elif problem == "flower":
        data_dir = "./output_turek_flower/"
        Var = Rad/5
        Obs_y = 0.2

    elif problem == "flower_symm":
        data_dir = "./output_turek_flower_symm/"
        Var = Rad/5
        Obs_y = Wid/2

    elif problem == "cylinder_symm_wide":
        data_dir = "./output_turek_cylinder_symm_wide/"
        Var = 0.0
        Wid = Wid*2
        Obs_y = Wid/2
        print(f"{Obs_y=}")

    elif problem == "cylinder_wide":
        data_dir = "./output_turek_cylinder_wide/"
        Var = 0.0
        Wid = Wid*2
        Obs_y = Wid*0.487805

    else:
        raise ValueError(f"Unknown problem: {problem}")

    Re_list = np.linspace(0,501,1003)
    Umax_list = np.array(Re_list)*3/2/100

    drag_values_list = []
    lift_values_list = []
    for Umax in Umax_list:
        #print(f"Re = {Umax/3*2*100:.1f}")

        folder_name = f"reynolds_nb_data/Re={Umax/3*2*100:.1f}"
        #print(f"Folder you want to access: {folder_name}")
        subdir_path = os.path.join(data_dir, folder_name)
        traction_file = f"{subdir_path}/traction_data.npz"

        
        traction_file = f"{subdir_path}/traction_data.npz"

        if os.path.exists(traction_file):
            data = np.load(traction_file)
            drag_coords = data['drag_coords']
            drag_values = data['drag_values'].tolist()
            lift_coords = data['lift_coords']
            lift_values = data['lift_values'].tolist()

        else:   
            raise FileNotFoundError(
                f"Traction data file not found: {traction_file}"
            )

        # Append to lists
        drag_values_list.append(drag_values)
        lift_values_list.append(lift_values)

    turning_points = [(85, 7), (33.3, 78), (45, 315)]
    turning_points_lift = [(np.deg2rad(theta), Re) for (theta, Re) in turning_points]
    turning_points_lift += [(-theta, Re) for (theta, Re) in turning_points_lift]
    turning_points = [(118, 0), (67.5, 48), (45.3,177), (55, 315), (108, 315), (163, 315)]
    turning_points_drag = [(np.deg2rad(theta), Re) for (theta, Re) in turning_points]
    turning_points_drag += [(-theta, Re) for (theta, Re) in turning_points_drag]


    subdir_path = os.path.join(data_dir, "traction_profiles")
    os.makedirs(subdir_path, exist_ok=True)
    drag_countours_name = f"{subdir_path}/contour_drag"
    lift_countours_name = f"{subdir_path}/contour_lift"
    plot_2d_contours_with_derivatives(lift_coords,lift_values_list,Re_list,r"pointwise lift [N/m] parametrisation as surface in $(\theta,Re)$ space",lift_countours_name, r"t_\text{lift}", turning_points_lift, Obs_y, problem)
    plot_2d_contours_with_derivatives(drag_coords,drag_values_list,Re_list,r"pointwise drag [N/m] parametrisation as surface in $(\theta,Re)$ space",drag_countours_name, r"t_\text{drag}", turning_points_drag, Obs_y, problem)
