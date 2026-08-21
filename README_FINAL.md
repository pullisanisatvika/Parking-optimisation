## Dynamic Parking Recommendation — Final Thesis Implementation

This repository contains the final validated thesis implementation and frozen experimental artifacts for the dynamic parking recommendation study.

### Final Objectives

Baseline objective:

`Z = Alpha * (external_drive_min + external_walk_min) + (1 - Alpha) * parking_cost`

Extension objective:

`Z = Alpha * (external_drive_min + internal_drive_min + internal_walk_min + external_walk_min) + (1 - Alpha) * parking_cost`

### Final Authoritative Result Files

- `parking-lot-recommendation-baseline/RESULTS/master_results_all_methods_for_graphs.csv`
- `parking-lot-recommendation-extension/RESULTS/COMBINED_ALL_METHODS/results_all_methods_spot_level.csv`

### Final Graphs

- `FINAL_GRAPHS/`
- Contains 44 final PDF figures and their preserved graph-source CSV files

### Algorithms

Baseline:

- Optimization
- Heuristic
- ACO
- PSO
- GA

Extension:

- ACO
- PSO
- GA
- Derived Optimization reference where applicable

### Repository Scope

The preserved CSVs and figures represent the final thesis results. The repository has been cleaned after final experimental validation while preserving production code, datasets, result tables, and graph-generation inputs required for reproduction.
