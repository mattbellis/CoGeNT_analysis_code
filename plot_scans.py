import numpy as np
import matplotlib.pylab as plt
import seaborn as sn
from scipy.integrate import trapz

import sys


# If batch
plt.switch_backend('Agg')

################################################################################
def calc90ul(x,diff):

    diff *= -1.0
    y = np.exp(diff)
    #print len(y),len(x)

    tot_area = trapz(y,x=x)

    #print "tot_area: ",tot_area
    
    partial_area = None
    ul = -1
    for i in range(0,len(diff)):
        partial_area = trapz(y[i:],x=x[i:])
        #print "\tpartial area: ",partial_area
        if partial_area<0.90*tot_area:
            ul = x[i]
            return ul

    return ul
            



################################################################################

filenames = sys.argv[1:]

tag = "default"
bkglh = 7817.210756300921
erange = "default"

if filenames[0].find('nicole')>=0:
    tag = "scans_nicole"
    if filenames[0].find('erange_0.50')>=0:
        bkglh = 7799.163552624375
        erange = "0.50-3.2"
    elif filenames[0].find('erange_0.55')>=0:
        bkglh = 7526.225683365021 # 0.55-3.2
        erange = "0.55-3.2"
elif filenames[0].find('juan')>=0:
    tag = "scans_juan"
    if filenames[0].find('erange_0.50')>=0:
        bkglh = 7817.210756300921
        erange = "0.50-3.2"
    elif filenames[0].find('erange_0.55')>=0:
        bkglh = 7543.610469393319 # 0.55-3.2
        erange = "0.55-3.2"

if filenames[0].find('stream')>=0:
    tag = "%s_stream" % (tag)

tag = "%s_%s" % (tag,erange)

mass = []
xsec = []
lh = []

for fn in filenames:
    print fn
    f = open(fn)
    x = f.readline()
    #print "----"
    #print x
    if x.find('nan')<0: # No nans
        totresults = eval(x)
        #print totresults
        a,b,c = totresults['final_values']['mDM'], totresults['final_values']['sigma_n'], totresults['final_values']['final_lh']
    
        if a==a and b==b and c==c:
            mass.append(a)
            xsec.append(b)
            lh.append(c)

lh = np.array(lh)
xsec = np.array(xsec)
mass = np.array(mass)

print lh
minlh = min(lh)

'''
plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
plt.gca().tick_params(axis='both', which='major', labelsize=18)
plt.plot(mass,lh-min(lh)+0.01,'o')
plt.ylabel(r'$\Delta \mathcal{L}$',fontsize=36)
plt.xlabel(r'Mass (GeV/c$^2$)',fontsize=24)
plt.yscale('log')

plt.subplot(1,2,2)
plt.gca().tick_params(axis='both', which='major', labelsize=18)
plt.plot(xsec,lh-min(lh)+0.01,'o')
plt.ylabel(r'$\Delta \mathcal{L}$',fontsize=36)
plt.xlabel(r'$\sigma_N$ (barns)',fontsize=24)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
'''


xsecvals = np.unique(xsec)
massvals = np.unique(mass)

print xsecvals

################################################################################
# Get the best values (lowest LH)
scanxseclh = np.zeros(len(xsecvals))
for i,x in enumerate(xsecvals):
    y = lh[xsec==x]
    scanxseclh[i] = min(y)


scanmasslh = np.zeros(len(massvals))
for i,x in enumerate(massvals):
    y = lh[mass==x]
    scanmasslh[i] = min(y)


for a in scanxseclh:
    print a,xsecvals[scanxseclh==a],massvals[scanmasslh==a]

orgbkglh = bkglh
bkglh -= min(lh)

massdiff = scanmasslh-min(lh)
xsecdiff = scanxseclh-min(lh)
################################################################################

################################################################################
# Get the ULs
ulbymass = np.zeros(len(massvals))
xulbymass = np.zeros(len(massvals))
print "MASSVALS"
print massvals
sortedmassvals = np.sort(massvals)
for i,x in enumerate(sortedmassvals):
    l = lh[mass==x]
    y = xsec[mass==x]

    l -= minlh

    l = l[y.argsort()]
    y.sort()
    #print l,y
    ul = calc90ul(y,l)
    print x,ul

    ulbymass[i] = ul
    xulbymass[i] = x


################################################################################

# Only plot the minima
plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
plt.gca().tick_params(axis='both', which='major', labelsize=18)
plt.plot(mass,lh-min(lh)+0.01,'o',alpha=0.2,markersize=10,label='All scans')
plt.plot(massvals,massdiff+0.01,'o',markersize=15,label=r'Best $\Delta \mathcal{L}$')
plt.plot([min(massvals),max(massvals)],[bkglh,bkglh],'k--',label='Background only')
plt.ylabel(r'$\Delta \mathcal{L}$',fontsize=36)
plt.xlabel(r'Mass (GeV/c$^2$)',fontsize=24)
plt.yscale('log')
plt.legend(loc='lower right',fontsize=18)

plt.subplot(1,2,2)
plt.gca().tick_params(axis='both', which='major', labelsize=18)
plt.plot(xsec,lh-min(lh)+0.01,'o',alpha=0.2,markersize=10,label='All scans')
plt.plot(xsecvals,xsecdiff+0.01,'o',markersize=15,label=r'Best $\Delta \mathcal{L}$')
plt.plot([min(xsecvals),max(xsecvals)],[bkglh,bkglh],'k--',label='Background only')
plt.ylabel(r'$\Delta \mathcal{L}$',fontsize=36)
plt.xlabel(r'$\sigma_N$ (cm$^2$)',fontsize=24)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left',fontsize=18)
plt.tight_layout()

name = "Plots/scan_results_%s.png" % (tag)
plt.savefig(name)



print xulbymass,ulbymass
plt.figure()
plt.plot(xulbymass,ulbymass,'o-')
plt.yscale('log')

filename = "upper_limits_%s.dat" % (tag)
outfile = open(filename,'w+')
for a,b in zip(xulbymass,ulbymass):
    output = "%f %e\n" % (a,b)
    outfile.write(output)
outfile.close()

################################################################################
# What is the significance?
################################################################################

import scipy.stats as stats

lh0 = orgbkglh
lh1 = minlh

delta_ndof = 2

D = 2*np.abs(lh0 - lh1)

sig = stats.chisqprob(D,delta_ndof)

# page 91 http://www.slac.stanford.edu/BFROOT/www/Statistics/Report/report.pdf
# I think this is D

#print "\n\n"
#print "D:   %f" % (D)
print "noWIMP/withWIMP/diff/D/sig: %f %f %f %f %f" % (lh0,lh1,(lh1-lh0),D,sig)


print orgbkglh
print minlh
print orgbkglh-minlh
sigma = np.sqrt(2*(orgbkglh-minlh))
print 'sigma: ',sigma

print erange
print tag
print minlh,mass[lh==minlh],xsec[lh==minlh]



#plt.show()

