
import numpy as np
import matplotlib.pyplot as plt

lambda_rate = 0.1  # expected number of hacks per day
mu = 10            #  meanof log losses 
sigma = 0.5         # variability of log losses
T = 30              # time period (days)
num_simulations = 10000000 # number of monte carlo simulations
total_losses = []



for _ in range(num_simulations):
    num_hacks = np.random.poisson(lambda_rate * T)

    if num_hacks > 0:
        losses = np.random.lognormal(mean=mu, sigma=sigma, size=num_hacks)
    else:
        losses = []

    total_loss = sum(losses)
    total_losses.append(total_loss)

plt.figure(figsize=(10, 6))
plt.hist(total_losses, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
plt.axvline(np.percentile(total_losses, 95), color="red", linestyle="dashed", linewidth=2, label="95% var")
plt.axvline(np.mean(total_losses), color="green", linestyle="dashed", linewidth=2, label="expected loss")
plt.title("simulation for freq. loss of web3 wallet hacking")
plt.xlabel("loss ($)")
plt.ylabel("freq.")
plt.legend()
plt.grid(True)
plt.show()

mean_loss = np.mean(total_losses)
var_95 = np.percentile(total_losses, 95)
cvar_95 = np.mean([loss for loss in total_losses if loss >= var_95])

mean_loss, var_95, cvar_95
