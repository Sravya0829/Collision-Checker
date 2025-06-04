# Collision-Checker
This project implements an efficient occupancy grid based system for robotic path planning and collision detection. Key features include:

Occupancy Grid Mapping: Constructed a 2D occupancy grid to represent the environment for fast collision queries.
Geometric Utilities: Developed vectorized functions to generate robot footprints and perform batched geometric transformations.
Path Validation: Implemented batched collision checking using NumPy for sampled robot poses, with linear interpolation to validate local paths between poses.
High-Performance Computation: Leveraged np.einsum, broadcasting, and NumPy vectorization for fast batched operations on robot configurations and geometry.
Visualization: Used Matplotlib to render occupancy maps, robot positions, and highlight collision free vs. obstructed paths.
