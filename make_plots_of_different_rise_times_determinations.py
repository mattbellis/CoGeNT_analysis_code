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

energies = np.linspace(0.5,3.2,1000)

################################################################################
# Fast
################################################################################

figs = []
axes = []

for i in range(0,1):
    figs.append(plt.figure(figsize=(8,8)))
    axes.append([])
    for j in range(0,9):
        axes[i].append(figs[i].add_subplot(3,3,j+1))

cols = ['red','blue']
for i in range(0,2):
    # Read in the rise-time parameters from the file passed in on the commandline
    rt_parameters_filename = None
    if i==0:
        rt_parameters_filename = 'risetime_parameters_from_data_data_constrained_with_simulated_Nicole.py'
    elif i==1:
        rt_parameters_filename = 'risetime_parameters_from_data_risetime_parameters_risetime_determination_juan.py'

    rt_parameters_filename = rt_parameters_filename.rstrip('.py')
    print("Rise-time parameters_filename: %s" % (rt_parameters_filename))
    rt_parameters_file = __import__(rt_parameters_filename)
    risetime_parameters = getattr(rt_parameters_file,'risetime_parameters')

    fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,mu0,sigma0,mu,sigma = risetime_parameters()

    params = [fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,mu0,sigma0,mu,sigma]

    expfunc = lambda p, x: p[1]*np.exp(-p[0]*x) + p[2]
    expfunc1 = lambda p, x: p[1]*x + p[0]

    for j,p in enumerate(params):

        if j<6:
            print(p)
            y = expfunc(p,energies)
            #print y
        else:
            print(p)
            y = expfunc1(p,energies)

        axes[0][j].plot(energies,y,color=cols[i])


plt.show()

