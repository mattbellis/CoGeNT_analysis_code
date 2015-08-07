from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

from cogent_utilities import rise_time_prob_fast_exp_dist,rise_time_prob_exp_progression
import parameters 

import lichen.pdfs as pdfs
import seaborn as sn

import sys

xbins = 500
ybins = 500

ranges,subranges,nbins = parameters.fitting_parameters(0)

results_file = open(sys.argv[1])
pars = eval(results_file.readline())

elo = ranges[0][0]
ehi = ranges[0][1]
print elo,ehi
tlo = ranges[1][0]
thi = 1238

E = np.linspace(elo,ehi,xbins) # energy
t = np.linspace(tlo,thi,ybins) # time

if sys.argv[2]=="0":
    # Surface events
    tag = "surface"
    label = "%s events" % (tag)
    plt.figure(figsize=(6,4))
    pdf  = pdfs.poly(E,[pars['k1_surf'],pars['k2_surf']],elo,ehi) #,efficiency=efficiency)
    plt.plot(E,pdf,'y-',linewidth=4,label=label)
    plt.xlabel(r'Energy (keVee)',fontsize=18)
    plt.ylabel('PDF (arbitrary units)',fontsize=18)
    plt.xlim(elo,ehi)
    plt.ylim(0)
    plt.tight_layout()
    plt.legend(fontsize=18)
    name = "Plots/pdf_%s_E.png" % (tag)
    plt.savefig(name)

    plt.figure(figsize=(6,4))
    pdf  = pdfs.exp(t,pars['t_surf'],tlo,thi)#,subranges=subranges[1])
    plt.plot(t,pdf,'y-',linewidth=4,label=label)
    plt.xlabel(r'Days since 12/3/2009',fontsize=18)
    plt.ylabel('PDF (arbitrary units)',fontsize=18)
    plt.xlim(tlo,thi)
    plt.ylim(0,0.002)
    plt.tight_layout()
    plt.legend(fontsize=18)
    name = "Plots/pdf_%s_t.png" % (tag)
    plt.savefig(name)


elif sys.argv[2]=="1":
    # Neutron events
    tag = "neutron"
    label = "%s events" % (tag)
    plt.figure(figsize=(6,4))
    pdf  =  pdfs.exp(E,pars['flat_neutrons_slope'],elo,ehi)
    plt.plot(E,pdf,'c-',linewidth=4,label=label)
    plt.xlabel(r'Energy (keVee)',fontsize=18)
    plt.ylabel('PDF (arbitrary units)',fontsize=18)
    plt.xlim(elo,ehi)
    plt.ylim(0)
    plt.legend(fontsize=18)
    plt.tight_layout()
    name = "Plots/pdf_%s_E.png" % (tag)
    plt.savefig(name)

    plt.figure(figsize=(6,4))
    pdf  =  pdfs.poly(t,[],tlo,thi)
    plt.plot(t,pdf,'c-',linewidth=4,label=label)
    plt.xlabel(r'Days since 12/3/2009',fontsize=18)
    plt.ylabel('PDF (arbitrary units)',fontsize=18)
    plt.xlim(tlo,thi)
    plt.ylim(0,0.002)
    plt.legend(fontsize=18)
    plt.tight_layout()
    name = "Plots/pdf_%s_t.png" % (tag)
    plt.savefig(name)



plt.show()

