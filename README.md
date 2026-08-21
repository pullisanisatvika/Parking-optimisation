# Parking Optimisation

An AI-powered parking recommendation and optimisation system that identifies suitable parking options while balancing parking cost, travel time, walking distance, and traffic conditions.

## Features

- Parking recommendation based on multiple optimisation objectives
- Ant Colony Optimisation (ACO) for route selection
- Particle Swarm Optimisation (PSO) and Genetic Algorithm (GA) comparisons
- SUMO-based traffic simulation
- Baseline and extended parking-spot experiments
- Result visualisations and validation reports

## Project Structure

- `parking-lot-recommendation-baseline/` - Baseline optimisation implementation
- `parking-lot-recommendation-extension/` - Extended parking-spot recommendation system
- `FINAL_GRAPHS/` - Final result figures and objective-value comparisons
- `verification_reports/` - Numerical verification outputs
- `aco_results/` - ACO experiment results
- `final_cases/` - Generated parking and traffic test cases

## Requirements

Python 3.10+ is recommended. Install dependencies from the relevant project folder:

```bash
pip install -r parking-lot-recommendation-extension/requirements.txt
