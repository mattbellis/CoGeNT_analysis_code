import numpy as np
import lichen.lichen as lch
import matplotlib.pyplot as plt

from datetime import datetime,timedelta,date
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from cogent_utilities import sec2days

import sys
import seaborn as sn

tag = "default"

vals = np.loadtxt(sys.argv[1])

seconds = vals[:,0]
days = sec2days(seconds)
energies = vals[:,1]
risetimes = vals[:,2]

elo = 0.5
ehi = 3.3
enbins = 70
ewidth = (ehi-elo)/enbins

dlo = 0.0
dhi = 1240.
dnbins = 62
dwidth = (dhi-dlo)/dnbins

rlo = 0.0
rhi = 5.0
rnbins = 120
rwidth = (rhi-rlo)/rnbins

index = np.ones(len(days)).astype(bool)

if sys.argv[2]=="0":
    index = days>0
    index *= days<1238
    index *= energies>elo
    index *= energies<ehi

elif sys.argv[2]=="1" or sys.argv[2]=="2":
    index = days>0
    index *= days<1238
    index *= energies>0.0
    index *= energies<12.0
    print("EHER")
    print(index)
    tag = "full_range"

    elo = 0.0
    ehi = 12.0
    enbins = 240
    ewidth = (ehi-elo)/enbins

    dlo = 0.0
    dhi = 1240.
    dnbins = 62
    dwidth = (dhi-dlo)/dnbins

    rlo = 0.0
    rhi = 10.0
    rnbins = 100
    rwidth = (rhi-rlo)/rnbins


print(days)
print(energies)
print(risetimes)

days = days[index]
energies = energies[index]
risetimes = risetimes[index]

print(days)


if sys.argv[2]=="0" or sys.argv[2]=="2":
    ################################################################################
    plt.figure(figsize=(8,4))
    lch.hist_err(energies,bins=enbins,linewidth=2,range=(elo,ehi))
    name = "# interactions/ %0.2f keVee" % (ewidth)
    plt.ylabel(name,fontsize=14)
    plt.xlabel('Recoil energy (keVee)',fontsize=18)
    plt.ylim(0)
    plt.xlim(elo,ehi)
    if sys.argv[2]=="2":
        plt.yscale('log')
        plt.ylim(20)
    plt.tight_layout()
    name = "Plots/cogent_data_energy_%s.png" % (tag)
    plt.savefig(name)

    ################################################################################
    plt.figure(figsize=(8,4))
    lch.hist_err(days,bins=dnbins,linewidth=2,range=(dlo,dhi))
    name = "# interactions/%d days" % (dwidth)
    plt.ylabel(name,fontsize=14)
    plt.xlabel('Days since 12/3/2009',fontsize=18)
    plt.ylim(0,300)
    if sys.argv[2]=="2":
        plt.ylim(0,1200)

    # Plot the date labels.
    start_date = date(2009,12,3)
    def todate(x, pos, today=start_date):
        return today+timedelta(days=x)
    fmt = ticker.FuncFormatter(todate)
    plt.gca().xaxis.set_major_formatter(fmt)
    plt.gcf().autofmt_xdate(rotation=45)
    plt.xlim(dlo,dhi)
    plt.tight_layout()
    name = "Plots/cogent_data_time_%s.png" % (tag)
    plt.savefig(name)

    ################################################################################
    plt.figure(figsize=(8,4))
    lch.hist_err(risetimes,bins=rnbins,linewidth=2,range=(rlo,rhi))
    name = r"# interactions/ %0.2f $\mu$s" % (rwidth)
    plt.ylabel(name,fontsize=14)
    plt.xlabel(r'Rise times ($\mu$s)',fontsize=18)
    plt.ylim(0)
    plt.xlim(rlo,rhi)
    if sys.argv[2]=="2":
        plt.yscale('log')
        plt.xlim(0.0,rhi)
        plt.ylim(0.1)
    plt.tight_layout()
    name = "Plots/cogent_data_risetime_%s.png" % (tag)
    plt.savefig(name)

elif sys.argv[2]=="1":
    ################################################################################
    plt.figure(figsize=(8,4))
    plt.plot(energies,risetimes,'ko',markersize=2,alpha=0.2)
    name = r"Rise time ($\mu$s)"
    plt.ylabel(name,fontsize=14)
    plt.xlabel('Recoil energy (keVee)',fontsize=18)
    plt.xlim(0,12)
    plt.ylim(0.1,20)
    plt.yscale('log')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    name = "Plots/cogent_data_rtvse_%s.png" % (tag)
    plt.savefig(name)

plt.show()


