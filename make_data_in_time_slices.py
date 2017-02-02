import numpy as np
import lichen.lichen as lch
import matplotlib.pyplot as plt

from datetime import datetime,timedelta,date
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from cogent_utilities import sec2days

import sys
import seaborn as sn

#from pympler.tracker import SummaryTracker
#tracker = SummaryTracker()
from mem_top import mem_top

mem_top()


vals = np.loadtxt(sys.argv[1])

seconds = vals[:,0]
org_days = sec2days(seconds)
org_energies = vals[:,1]
org_risetimes = vals[:,2]

elo = 0.5
ehi = 3.3
enbins = 70
ewidth = (ehi-elo)/enbins

dlo = 0.0
#dhi = 35.
dhi = 1240.
dnbins = 62
dwidth = (dhi-dlo)/dnbins

rlo = 0.0
rhi = 5.0
rnbins = 120
rwidth = (rhi-rlo)/rnbins

index = np.ones(len(org_days)).astype(bool)

# Discrete chunks
#ndays = 30
#nslices = int(dhi/ndays) + 1
#tag = "discrete"

# Sliding window
ndays = 30
nslices = int(dhi)
tag = "sliding"

for i in range(0,nslices):

    # Discrete chunks
    #daylo = ndays*i
    #dayhi = ndays*(i + 1)

    # Sliding window
    daylo = i
    dayhi = ndays + i

    print daylo
    print dayhi

    if sys.argv[2]=="0":
        index = org_days>daylo
        index *= org_days<dayhi
        index *= org_energies>elo
        index *= org_energies<ehi

    elif sys.argv[2]=="1" or sys.argv[2]=="2":
        index = org_days>daylo
        index *= org_days<dayhi
        index *= org_energies>0.0
        index *= org_energies<12.0
        #print index
        tag += "_full_range"

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


    #print days
    #print energies
    #print risetimes

    days = org_days[index]
    energies = org_energies[index]
    risetimes = org_risetimes[index]

    #print days

    if sys.argv[2]=="0" or sys.argv[2]=="2":
        ################################################################################
        fig = plt.figure(figsize=(8,4))
        ret,xpts,ypts,xpts_err,ypts_err = lch.hist_err(energies,bins=enbins,linewidth=2,range=(elo,ehi))
        name = "# interactions/ %0.2f keVee" % (ewidth)
        plt.ylabel(name,fontsize=14)
        plt.xlabel('Recoil energy (keVee)',fontsize=18)
        plt.ylim(0,25) # This might have to be changed for sliding or discrete
        plt.xlim(elo,ehi)
        if sys.argv[2]=="2":
            plt.yscale('log')
            plt.ylim(20)
        plt.tight_layout()
        name = "animation_plots/cogent_data_energy_%s_%04d.png" % (tag,i)
        plt.savefig(name)
        del ret,xpts,ypts,xpts_err,ypts_err 
        

        ################################################################################
        fig = plt.figure(figsize=(8,4))
        ret,xpts,ypts,xpts_err,ypts_err = lch.hist_err(days,bins=dnbins,linewidth=2,range=(dlo,dhi))
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
        name = "animation_plots/cogent_data_time_%s_%04d.png" % (tag,i)
        plt.savefig(name)
        del ret,xpts,ypts,xpts_err,ypts_err 
        

        ################################################################################
        fig = plt.figure(figsize=(8,4))
        ret,xpts,ypts,xpts_err,ypts_err = lch.hist_err(risetimes,bins=rnbins,linewidth=2,range=(rlo,rhi))
        name = r"# interactions/ %0.2f $\mu$s" % (rwidth)
        plt.ylabel(name,fontsize=14)
        plt.xlabel(r'Rise times ($\mu$s)',fontsize=18)
        plt.ylim(0,25) # This might have to be changed for sliding or discrete
        plt.xlim(rlo,rhi)
        if sys.argv[2]=="2":
            plt.yscale('log')
            plt.xlim(0.0,rhi)
            plt.ylim(0.1)
        plt.tight_layout()
        name = "animation_plots/cogent_data_risetime_%s_%04d.png" % (tag,i)
        plt.savefig(name)
        del ret,xpts,ypts,xpts_err,ypts_err 
        plt.close('all')

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
        name = "animation_plots/cogent_data_rtvse_%s_%04d.png" % (tag,i)
        plt.savefig(name)

    del days
    del energies
    del risetimes
    del index

#plt.show()

#tracker.print_diff()
