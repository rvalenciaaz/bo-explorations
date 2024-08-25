import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

print(f"Running on PyMC v{pm.__version__}")

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

d = pd.read_csv(
    "https://raw.githubusercontent.com/pymc-devs/resources/master/Rethinking_2/Data/milk.csv",
    sep=";",
)
d = d[["kcal.per.g", "neocortex.perc", "mass"]].rename({"neocortex.perc": "neocortex"}, axis=1)
d["log_mass"] = np.log(d["mass"])
d = d[~d.isna().any(axis=1)].drop("mass", axis=1)
d.iloc[:, 1:] = d.iloc[:, 1:] - d.iloc[:, 1:].mean()
print(d.head())

with pm.Model() as model_0:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", 10)

    mu = alpha + beta * d["neocortex"]

    kcal = pm.Normal("kcal", mu=mu, sigma=sigma, observed=d["kcal.per.g"])
    trace_0 = pm.sample(2000, return_inferencedata=True)

with pm.Model() as model_1:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", 10)

    mu = alpha + beta * d["log_mass"]

    kcal = pm.Normal("kcal", mu=mu, sigma=sigma, observed=d["kcal.per.g"])

    trace_1 = pm.sample(2000, return_inferencedata=True)

with pm.Model() as model_2:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
    sigma = pm.HalfNormal("sigma", 10)

    mu = alpha + pm.math.dot(beta, d[["neocortex", "log_mass"]].T)

    kcal = pm.Normal("kcal", mu=mu, sigma=sigma, observed=d["kcal.per.g"])

    trace_2 = pm.sample(2000, return_inferencedata=True)

traces = [trace_0, trace_1, trace_2]
az.plot_forest(traces, figsize=(10, 5))

plt.show()
plt.close()

ax = az.plot_density(
    traces,
    var_names=["alpha", "sigma"],
    shade=0.1,
    data_labels=["Model 0 (neocortex)", "Model 1 (log_mass)", "Model 2 (neocortex+log_mass)"],
)

ax[0, 0].set_xlabel("Density")
ax[0, 0].set_ylabel("")
ax[0, 0].set_title("95% Credible Intervals: alpha")

ax[0, 1].set_xlabel("Density")
ax[0, 1].set_ylabel("")
ax[0, 1].set_title("95% Credible Intervals: sigma")

plt.show()
plt.close()

print(traces[0])
#problem here
model_dict = dict(zip(["model_0", "model_1", "model_2"], traces))
comp = az.compare(model_dict)

ppc_w = pm.weight_predictions(
    traces=traces,
    models=[model_0, model_1, model_2],
    weights=comp.weight.sort_index(ascending=True)
    #progressbar=True,
)


ppc_2 = pm.sample_posterior_predictive(trace=trace_2, model=model_2, progressbar=False)



mean_w = ppc_w["kcal"].mean()
hpd_w = az.hdi(ppc_w["kcal"].flatten())

mean = ppc_2["kcal"].mean()
hpd = az.hdi(ppc_2["kcal"].flatten())

plt.plot(mean_w, 1, "C0o", label="weighted models")
plt.hlines(1, *hpd_w, "C0")
plt.plot(mean, 0, "C1o", label="model 2")
plt.hlines(0, *hpd, "C1")

plt.yticks([])
plt.ylim(-1, 2)
plt.xlabel("kcal per g")
plt.legend()

plt.show()