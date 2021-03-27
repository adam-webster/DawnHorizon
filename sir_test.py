import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 50*50
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 20, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, nu, (in 1/days).
beta, nu = 15/100, 1./100
# time list
t = np.linspace(0, 200, 200)

# The SIR model differential equations.
def deriv(y, t, N, beta, nu):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - nu * I
    dRdt = nu * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over time
ret = odeint(deriv, y0, t, args=(N, beta, nu))
S, I, R = ret.T


# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.style.use('ggplot') # pretty format
plt.figure(figsize=(9,6))
plt.plot(t, S/N, 'b', lw=0.7, label='Susceptible')
plt.plot(t, I/N, 'r', lw=0.7, label='Infected')
plt.plot(t, R/N, 'g', lw=0.7, label='Recovered with immunity')
plt.title(f'SIR Model \n Population Size = {N}\n Beta = {beta:.4f}  Nu = {nu:.4f}   R0 = {beta/nu:.4f}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Frequency of individual')
plt.xlim(0, len(t))

plt.show()
