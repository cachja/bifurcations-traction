# Appendix C supporting code
Computation of turning-point maps for variations of the Schäfer--Turek benchmark geometry and obstacle shapes.

## Usage
Select a geometry by setting: `problem = problem_list[i]` for any element of `problem_list = ["cylinder","cylinder_symm","flower","flower_symm","cylinder_symm_wide","cylinder_wide"]`. This selection must be made consistently in both scripts: `traction_turning_point_map.py` and `compute_flow_and_traction.py`.

## Workflow
Run `python compute_flow_and_traction.py`. This generates traction data for the selected geometry in the corresponding output folder.

Then run `python traction_turning_point_map.py`. This produces a PDF file with the turning-point maps in the same folder.

## Output
- Final PDF figures are included in the repository
- Intermediate data files are not stored due to size considerations