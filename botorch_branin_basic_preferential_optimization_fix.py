#working with torch tensors
import torch

#from botorch.models import SingleTaskGP
from botorch.models import PairwiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
#from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption, qExpectedUtilityOfBestOption
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.optim import optimize_acqf

from botorch.utils.sampling import draw_sobol_samples
from torch.quasirandom import SobolEngine

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import seaborn as sns

import pandas as pd

import itertools



# Define the Branin function with input scaling, i.e, takes between [0,1] x [0,1] instead of the typical [0,15] x [-5,10]
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
#checking cuda and torch type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dtype = torch.float64

#design space bounds
bounds = torch.tensor([[0., 0.], [1., 1.]], dtype=dtype, device=device)


N = 7
batch_size=7
instances=20
iteration_number=20
iterations = list(range(1, iteration_number+1))
supra_best=[]

def comp_function(y):
    rlist=[]
    for pairs in itertools.combinations(y,2):
        indexleft=(y == pairs[0]).nonzero(as_tuple=True)[0].cpu().numpy()[0]
        indexright=(y == pairs[1]).nonzero(as_tuple=True)[0].cpu().numpy()[0]
        if pairs[0]>pairs[1]:
            rlist.append([indexleft,indexright])
        else:
            rlist.append([indexright, indexleft])
    return torch.tensor(rlist).to(device=device, dtype=dtype)


#instances are full B0 cycles, the algorithm is the same, but the initial Sobol sampling changes since no seed is defined.
#averaging over the instances will allow to quantify the performance of the BO
for ins in range(0,instances-1):

    print("Starting instance "+str(ins)+" ...")

    #sobol sampling for the initial samples
    sobol_engine = SobolEngine(dimension=2, scramble=False)  # 2 dimensions for your input space
    train_x = draw_sobol_samples(bounds=bounds, n=1, q=N).squeeze(0)
    #print(train_x)

    #evaluate the initial samples in the known Branin function, in a real world problem this would be the expensive experiment
    #on preferential B0, this function could be more general, like a molecule plot or the final taste of a juice combination, a concept which is difficult to score
    #on this case we will keep it numerical, and the comparison function will be the biggest number wins in the pair
    train_y = branin(train_x, negate=True).unsqueeze(-1)

    print(train_y)
    comp_list=comp_function(train_y)
    print(comp_list)
    #a list to save the models after each iteration
    models = []

    
    #gp_model = SingleTaskGP(train_x, train_y).to(device=device, dtype=dtype)
    gp_model = PairwiseGP(train_x, comp_list).to(device=device, dtype=dtype)
    mll = PairwiseLaplaceMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_model(mll)

    #save the model
    models.append(gp_model)


    # To store data for animation
    frames_x = [train_x.cpu().numpy()]
    frames_y = [train_y.cpu().numpy()]
    best_points = []
    best_y_values=[]

    for iteration in range(iteration_number):
        EUBO= qExpectedUtilityOfBestOption(pref_model=gp_model)
        #EI = qExpectedImprovement(model=gp_model, best_f=train_y.max())#, maximize=True)
        candidate, _ = optimize_acqf(
            acq_function=EUBO,
            bounds=bounds,
            q=batch_size,
            num_restarts=5,
            raw_samples=20,
            options={"dtype": dtype, "device": device}
        )
        
        new_y = branin(candidate, negate=True).unsqueeze(-1)
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])
        
        comp_list=comp_function(train_y)

        gp_model = PairwiseGP(train_x, comp_list).to(device=device, dtype=dtype)
        mll =PairwiseLaplaceMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_model(mll)
        
        frames_x.append(train_x.cpu().numpy())
        frames_y.append(train_y.cpu().numpy())
        best_points.append(train_x[train_y.argmax(), :].cpu().numpy())
        best_y_values.append(train_y.max().cpu().numpy())
        models.append(gp_model)

    best_point = train_x[train_y.argmax(), :]
    best_value = train_y.max().item()

    best_y_values = np.array([element for element in best_y_values])
    supra_best.append(best_y_values)
    #torch.save(gp_model.state_dict())
    print("Best observed point:", best_point.cpu().numpy(), "Best observed value:", best_value)

    # Function to create the contour plot of the Branin function
    '''
    def plot_gp_mean(model, bounds, resolution=100):
        x1 = torch.linspace(bounds[0, 0], bounds[1, 0], resolution, dtype=dtype, device=device)
        x2 = torch.linspace(bounds[0, 1], bounds[1, 1], resolution, dtype=dtype, device=device)
        X1, X2 = torch.meshgrid(x1, x2)
        grid = torch.stack([X1.flatten(), X2.flatten()], -1)
        with torch.no_grad():
            mean = model.posterior(grid).mean.cpu().numpy().reshape(resolution, resolution)
        return X1.cpu().numpy(), X2.cpu().numpy(), mean

    # Update function for the animation
    def update(frame):
        plt.clf()
        X1, X2, mean = plot_gp_mean(models[frame], bounds)
        cp = plt.contourf(X1, X2, mean, levels=50, cmap=cm.viridis)
        plt.colorbar(cp)
        plt.scatter(frames_x[frame][:, 0], frames_x[frame][:, 1], color="red")
        plt.title(f"Iteration {frame+1}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ani = FuncAnimation(fig, update, frames=range(len(frames_x)), repeat=False)

    # Save the animation
    ani.save("branin_preferential_q_"+str(batch_size)+"_ins_"+str(ins)+".mp4", writer="ffmpeg", dpi=200)

    plt.close()
    print("Animation saved as branin_optimization_models.mp4")
    '''
    # Plotting the best Y values vs iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, best_y_values, marker='o', linestyle='-', color='b')
    plt.title('Best Y Values vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("best_value_vs_iterations_preferential_q_"+str(batch_size)+"_ins_"+str(ins)+".png",bbox_inches="tight",dpi=600)
    plt.close()

# Transposing 'supra_best' to get a list of 101 points for each of the 20 instances
#print(supra_best)
melted_data = []

# Iterate through each array and its index (using enumerate for iteration count)
for iteration, arr in enumerate(supra_best, start=1):
    for idx, value in enumerate(arr, start=1):
        melted_data.append({'Iteration': idx, 'Value': value, 'Array_Instance': iteration})

# Convert the list to a DataFrame
df_melted = pd.DataFrame(melted_data)

df_melted.to_csv("instances_preferential.csv",index=False)