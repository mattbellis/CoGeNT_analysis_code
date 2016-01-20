import numpy as np
import matplotlib.pylab as plt

import sys

filenames = sys.argv[1:]

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

plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
plt.plot(mass,lh-min(lh)+0.01,'o')
plt.ylabel(r'$\Delta \mathcal{L}$',fontsize=24)
plt.xlabel(r'Mass (GeV/c$^2$',fontsize=24)
plt.yscale('log')

plt.subplot(1,2,2)
plt.plot(xsec,lh-min(lh)+0.01,'o')
plt.ylabel(r'$\Delta \mathcal{L}$',fontsize=24)
plt.xlabel(r'$\sigma_N$ (barns)',fontsize=24)
plt.xscale('log')
plt.yscale('log')

plt.tight_layout()

plt.show()

