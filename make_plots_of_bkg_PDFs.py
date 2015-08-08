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

Eylim = [0,0]
tylim = [0,0]

fmt = None
Epdf = None
tpdf = None
label = None
tag = None

if sys.argv[2]=="0":
    # Surface events
    tag = "surface"
    label = "%s events" % (tag)
    fmt='y-'
    Epdf  = pdfs.poly(E,[pars['k1_surf'],pars['k2_surf']],elo,ehi) #,efficiency=efficiency)
    tpdf  = pdfs.exp(t,pars['t_surf'],tlo,thi)#,subranges=subranges[1])
    Eylim[0] = 0;Eylim[1] = None
    tylim[0] = 0;tylim[1] = 0.002


elif sys.argv[2]=="1":
    # Neutron events
    tag = "neutron"
    label = "%s events" % (tag)
    fmt = 'c-'
    Epdf  =  pdfs.exp(E,pars['flat_neutrons_slope'],elo,ehi)
    tpdf  =  pdfs.poly(t,[],tlo,thi)
    Eylim[0] = 0;Eylim[1] = None
    tylim[0] = 0;tylim[1] = 0.0014

elif sys.argv[2]=="2":
    # Compton events
    tag = "Compton"
    label = "%s events" % (tag)
    fmt = 'm-'
    Epdf  =  pdfs.exp(E,pars['e_exp_flat'],elo,ehi)
    tpdf  =  pdfs.exp(t,pars['t_exp_flat'],tlo,thi)
    Eylim[0] = 0;Eylim[1] = 0.60
    tylim[0] = 0;tylim[1] = 0.0015


plt.figure(figsize=(6,4))
plt.plot(E,Epdf,fmt,linewidth=4,label=label)
plt.xlabel(r'Energy (keVee)',fontsize=18)
plt.ylabel('PDF (arbitrary units)',fontsize=18)
plt.xlim(elo,ehi)
plt.ylim(Eylim[0],Eylim[1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=18)
plt.tight_layout()
name = "Plots/pdf_%s_E.png" % (tag)
plt.savefig(name)

plt.figure(figsize=(6,4))
plt.plot(t,tpdf,fmt,linewidth=4,label=label)
plt.xlabel(r'Days since 12/3/2009',fontsize=18)
plt.ylabel('PDF (arbitrary units)',fontsize=18)
plt.xlim(tlo,thi)
plt.ylim(tylim[0],tylim[1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=18)
plt.tight_layout()
name = "Plots/pdf_%s_t.png" % (tag)
plt.savefig(name)


plt.show()

