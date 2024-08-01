README.txt for Non_Intrusive_RB_MT Project
==========================================

Overview
--------
The Non_Intrusive_RB_MT project aims to find a reduced basis representation for an MT (Magnetotellurics) forward operator. 
The project involves generating random models, 
calculating snapshots using a forward operator, 
applying normalization, 
creating a basis for reduced-order modeling, 
calculating coefficients for projection onto the basis, 
and exporting significant matrices for further analysis.

Prerequisites
-------------
- Prepared on Julia (Version 1.9.3)
- Julia packages: Distributions, Turing, Plots, Surrogates, LinearAlgebra, CSV, DataFrames

Usage
-----
The project is structured into scripts that should be run sequentially:

01_generate_random_models.jl: Generates matrices of random models and models with a smoothness constraint.
02_generate_snapshots.jl: Uses the forward operator to create snapshot matrices from the models.
03_min_max_norm_snapshots.jl: Applies normalization to the snapshot matrices.
04_basis_creation.jl: Performs Singular Value Decomposition (SVD) to create a basis for reduced-order modeling from the normalized snapshots.
05_coefficient_calculation.jl: Calculates coefficients for projecting the snapshots onto the reduced basis.
06_matrix_exports.jl: Exports all generated matrices to CSV files for further analysis.

Scripts are located in the scripts/ directory and should be run from the project's root directory to ensure relative paths are correctly resolved.

Directory Structure
-------------------
/scripts: Contains Julia scripts for each step of the process.
/src: Includes source code defining the MT forward operator and normalization procedures.
/test: Destination for exported CSV files.

Exported Data
-------------
M1.csv and M2.csv: Original and smoothly constrained model matrices.
S1.csv and S2.csv: Snapshot matrices from random and smooth models.
S1_norm.csv and S2_norm.csv: Normalized snapshot matrices.
U1.csv and U2.csv: Bases created from SVD of the normalized snapshots.
A1.csv and A2.csv: Coefficients for projecting snapshots onto the bases.
