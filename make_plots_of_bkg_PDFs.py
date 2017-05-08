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
print(elo,ehi)
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
    tag = r"$n$ and $\alpha$"
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

elif sys.argv[2]=="3":
    # Compton events
    tag = "L-shell"
    label = "%s events" % (tag)
    fmt = 'r-'
    means = []
    sigmas = []
    numls = []
    decay_constants = []
    for i in range(11):
        name = "ls_mean%d" % (i)
        means.append(pars[name])
        name = "ls_sigma%d" % (i)
        sigmas.append(pars[name])
        name = "ls_ncalc%d" % (i)
        #numls.append(pars[name]/num_tot) # Normalized this # to number of events.
        numls.append(pars[name])
        name = "ls_dc%d" % (i)
        decay_constants.append(pars[name])

    Epdf = np.zeros(len(E))
    tpdf = np.zeros(len(t))
    #figE = plt.figure("figE",figsize=(6,4))
    #figt = plt.figure("figt",figsize=(6,4))
    fig = plt.figure("fig",figsize=(6,8))
    for n,m,s,dc in zip(numls,means,sigmas,decay_constants):
        print(n)
        Epdf_temp = n*pdfs.gauss(E,m,s,elo,ehi)
        #print E,Epdf_temp
        tpdf_temp = n*pdfs.exp(t,-dc,tlo,thi)

        Epdf += Epdf_temp
        tpdf += tpdf_temp

        #plt.figure("figE")
        plt.figure("fig")
        plt.subplot(2,1,1)
        plt.plot(E,Epdf_temp,'r--',linewidth=2)
        #plt.figure("figt")
        plt.subplot(2,1,2)
        plt.plot(t,tpdf_temp,'r--',linewidth=2)
        Eylim[0] = 0;Eylim[1] = 3500
        tylim[0] = 0.001;tylim[1] = 5.0


if sys.argv[2]!="3":
    #plt.figure(figsize=(6,4))
    fig = plt.figure(figsize=(6,8))
    plt.subplot(2,1,1)
else:
    #plt.figure("figE")
    plt.figure("fig")
    plt.subplot(2,1,1)
plt.plot(E,Epdf,fmt,linewidth=4,label=label)
plt.xlabel(r'Energy (keVee)',fontsize=18)
plt.ylabel('PDF (arbitrary units)',fontsize=18)
plt.xlim(elo,ehi)
if sys.argv[2]=="3":
    plt.xlim(0.5,1.6)
plt.ylim(Eylim[0],Eylim[1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

if sys.argv[2]=="3":
    plt.legend(fontsize=18,loc='upper left')
    plt.yscale('log')
    plt.ylim(1,5000)
else:
    plt.legend(fontsize=18,loc='upper right')
plt.gca().get_yaxis().set_ticks([])
plt.tight_layout()
#name = "Plots/pdf_%s_E.png" % (tag)
name = "Plots/pdf_%s_both.png" % (tag)
plt.savefig(name)

if sys.argv[2]!="3":
    #plt.figure(figsize=(6,4))
    #plt.figure(figsize=(6,8))
    #plt.figure("fig")
    plt.subplot(2,1,2)
else:
    #plt.figure("figt")
    plt.figure("fig")
    plt.subplot(2,1,2)
    plt.yscale('log')
plt.plot(t,tpdf,fmt,linewidth=4,label=label)
plt.xlabel(r'Days since 12/3/2009',fontsize=18)
plt.ylabel('PDF (arbitrary units)',fontsize=18)
plt.xlim(tlo,thi)
plt.ylim(tylim[0],tylim[1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=18)
plt.gca().get_yaxis().set_ticks([])
plt.tight_layout()
#name = "Plots/pdf_%s_t.png" % (tag)
name = "Plots/pdf_%s_both.png" % (tag)
plt.savefig(name)


plt.show()

