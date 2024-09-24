import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
import numpy as np
import pandas as pd
from botorch.utils.sampling import draw_sobol_samples
from torch.quasirandom import SobolEngine
import itertools
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the Branin function with input scaling
def branin(x, negate=False):
    a = 1.0
    b = 5.1 / (4 * torch.pi**2)
    c = 5 / torch.pi
    d = 6
    e = 10
    f = 1 / (8 * torch.pi)
    
    x1 = 15 * x[:, 0] - 5
    x2 = 15 * x[:, 1]
    
    result = a * (x2 - b * x1**2 + c * x1 - d)**2 + e * (1 - f) * torch.cos(x1) + e
    
    if negate:
        return -result
    else:
        return result

# Function to compute the maximum value of the negated Branin function
def compute_max_branin_negate(bounds, resolution=1000, device='cpu', dtype=torch.float64):
    x1 = torch.linspace(bounds[0, 0], bounds[1, 0], resolution, dtype=dtype, device=device)
    x2 = torch.linspace(bounds[0, 1], bounds[1, 1], resolution, dtype=dtype, device=device)
    X1, X2 = torch.meshgrid(x1, x2)
    grid = torch.stack([X1.flatten(), X2.flatten()], -1)
    with torch.no_grad():
        branin_values = branin(grid, negate=True)
    max_y = branin_values.max().item()
    return max_y

# Function to perform a single Bayesian Optimization run
def bo_run(batch_size, initial_samples, bounds, max_y, device, dtype, max_iterations=100):
    # Initialize with Sobol samples
    sobol_engine = SobolEngine(dimension=2, scramble=True)
    train_x = draw_sobol_samples(bounds=bounds, n=1, q=initial_samples, seed=torch.randint(0, 10000, (1,)).item()).squeeze(0)
    train_y = branin(train_x, negate=True).unsqueeze(-1)
    
    # Fit the GP model
    gp_model = SingleTaskGP(train_x, train_y).to(device=device, dtype=dtype)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_model(mll)
    
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        # Define the acquisition function
        EI = qExpectedImprovement(model=gp_model, best_f=train_y.max())
        
        # Optimize the acquisition function to find the next candidates
        try:
            candidate, _ = optimize_acqf(
                acq_function=EI,
                bounds=bounds,
                q=batch_size,
                num_restarts=5,
                raw_samples=20,
                options={"dtype": dtype, "device": device}
            )
        except Exception as e:
            print(f"Optimization failed at iteration {iterations}: {e}")
            break
        
        # Evaluate the objective function at the new candidates
        new_y = branin(candidate, negate=True).unsqueeze(-1)
        
        # Update the training data
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])
        
        # Refit the GP model
        gp_model = SingleTaskGP(train_x, train_y).to(device=device, dtype=dtype)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_model(mll)
        
        # Check the stopping criterion
        current_best = train_y.max().item()
        if current_best >= -0.5:
            break
    
    return iterations

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dtype, device=device)
    
    # Compute the maximum Branin value (negated) over a dense grid
    print("Computing the maximum negated Branin value over a dense grid...")
    max_y = compute_max_branin_negate(bounds, resolution=1000, device=device, dtype=dtype)
    print(f"Maximum negated Branin value (max_y): {max_y:.6f}")
    
    # Define the range for batch sizes and initial samples
    batch_sizes = range(1, 11)           # 1 to 10
    initial_samples_range = range(2, 11)  # 2 to 10
    runs_per_combination = 6
    
    # Prepare to collect results
    results = []
    
    # Iterate over all combinations of batch_size and initial_samples
    for batch_size, initial_samples in itertools.product(batch_sizes, initial_samples_range):
        # Iterate over the number of replicates for each combination
        for run in range(1, runs_per_combination + 1):
            iterations = bo_run(
                batch_size=batch_size,
                initial_samples=initial_samples,
                bounds=bounds,
                max_y=max_y,
                device=device,
                dtype=dtype,
                max_iterations=20
            )
            
            # Save each run's result individually
            results.append({
                'batch_size': batch_size,
                'initial_samples': initial_samples,
                'replicate': run,
                'iterations': iterations
            })
    
    # Create a DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("iterations_per_run.csv", index=False)
    print("\nAll results have been saved to 'iterations_per_run.csv'.")

if __name__ == "__main__":
    main()
