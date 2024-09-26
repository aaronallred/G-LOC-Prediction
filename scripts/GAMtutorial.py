from pygam import GAM, s, f
from pygam.datasets import wage
import matplotlib.pyplot as plt

X, y = wage(return_X_y=True)

## model
gam = GAM(s(0, n_splines=5) + s(1) + f(2) + s(3), distribution= 'binomial', link= 'logit')
gam.fit(X, y)

## plotting
plt.figure();
fig, axs = plt.subplots(1,1);
i = 0
XX = gam.generate_X_grid(term=i)
ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
if i == 0:
    ax.set_ylim(-30,30)

plt.show()