from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

from cogent_utilities import rise_time_prob_fast_exp_dist,rise_time_prob_exp_progression
import parameters 

import lichen.pdfs as pdfs
import seaborn as sn

xbins = 50
ybins = 50

X = np.linspace(0,5.0,xbins)
Y = np.linspace(0.5,3,ybins)

Zf = np.zeros((xbins,ybins))
Zs = np.zeros((xbins,ybins))

ranges,subranges,nbins = parameters.fitting_parameters(0)
ranges[2][0] = 0.0
ranges[2][1] = 5.0

################################################################################
# Fast
################################################################################
fast_mean_rel_k = [0.431998,-1.525604,-0.024960]
fast_sigma_rel_k = [-0.014644,5.745791,-6.168695]
fast_num_rel_k = [-0.261322,5.553102,-5.9144]

mu0 = [0.374145,0.628990,-1.369876]
sigma0 = [1.383249,0.495044,0.263360]

for j,y in enumerate(Y):
    for i,x in enumerate(X):
        Zf[j][i] = rise_time_prob_fast_exp_dist(x,y,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])
    Zf[j] /= max(Zf[j])

# Slow
# Using Nicole's simulated stuff
mu = [0.269108,0.747275,0.068146]
sigma = [0.531530,-0.020523]

for j,y in enumerate(Y):
    for i,x in enumerate(X):
        Zs[j][i] = rise_time_prob_exp_progression(x,y,mu,sigma,ranges[2][0],ranges[2][1])
    Zs[j] /= max(Zs[j])
################################################################################

X,Y = np.meshgrid(X, Y)
print X
print Y



# Plot the functions.
print len(X)
print len(Y)
print len(Zf)
figf = plt.figure(figsize=(8,4))
axf = figf.gca(projection='3d')
surff = axf.plot_wireframe(X, Y, Zf, rstride=1, cstride=1, linewidth=0.6, color='red')
axf.zaxis.set_major_locator(LinearLocator(10))
axf.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
axf.set_xlabel(r'Rise time ($\mu$s)',fontsize=18)
axf.set_ylabel('Energy (keVee)',fontsize=18)
plt.tight_layout()
plt.savefig("Plots/pdf_fast_rise_times.png")


figs = plt.figure(figsize=(8,4))
axs = figs.gca(projection='3d')
surfs = axs.plot_wireframe(X, Y, Zs, rstride=1, cstride=1, linewidth=0.3)
axs.zaxis.set_major_locator(LinearLocator(10))
axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
axs.set_xlabel(r'Rise time ($\mu$s)',fontsize=18)
axs.set_ylabel('Energy (keVee)',fontsize=18)
plt.tight_layout()
plt.savefig("Plots/pdf_slow_rise_times.png")


plt.show()

