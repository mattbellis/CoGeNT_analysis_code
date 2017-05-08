import numpy as np
import matplotlib.pylab as plt
import seaborn as sn
from scipy.integrate import trapz

import sys

################################################################################
def find_closest(x,val=5.0):

    closest = -999
    closest_index = -1
    min_diff = 1e7
    for i,a in enumerate(x):

        diff = abs(a-val)
        if diff<min_diff:
            min_diff = diff
            closest = a
            closest_index = i

    return closest,closest_index

    
################################################################################

def plot_contours_for_scans():
    filenames = sys.argv[1:]

    mass = []
    xsec = []
    lh = []

    sigma1mass = []
    sigma1xsec = []

    for fn in filenames:
        print(fn)
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

    print(lh)
    minlh = min(lh)

    xsecvals = np.unique(xsec)
    massvals = np.unique(mass)

    sigma1mass = []
    sigma1xsec = []
    sigma2mass = []
    sigma2xsec = []

    # Sig = sqrt(2*Dll)
    # Sig = 2
    # 4 = 2*Dll
    # Sig = 3
    # 9 = 2*Dll

    '''
    for m,x,l in zip(mass,xsec,lh):
        if l-minlh>1.8 and l-minlh<2.2:
            sigma1mass.append(m)
            sigma1xsec.append(x)
        if minlh-l>1.8 and minlh-l<2.2:
            sigma1mass.append(m)
            sigma1xsec.append(x)
        if l-minlh>4.2 and l-minlh<4.8:
            sigma2mass.append(m)
            sigma2xsec.append(x)
        if minlh-l>4.2 and minlh-l<4.8:
            sigma2mass.append(m)
            sigma2xsec.append(x)
    '''

    for m in massvals:

        index = mass==m

        temp_xsec = xsec[index]

        c,ci = find_closest(lh[index]-minlh,2.0)
        #print c,ci

        if abs(c-2.0)<0.2:

            sigma1mass.append(m)
            sigma1xsec.append(temp_xsec[ci])

        c,ci = find_closest(lh[index]-minlh,4.5)

        if abs(c-4.5)<0.2:

            sigma2mass.append(m)
            sigma2xsec.append(temp_xsec[ci])

        





    print(xsecvals)
    print(massvals)

    ################################################################################
    # Get the best values (lowest LH)
    bestmass = mass[lh==minlh]
    bestxsec = xsec[lh==minlh]

    ################################################################################

    # Only plot the minima
    plt.figure(figsize=(15,7))
    plt.subplot(1,1,1)
    plt.gca().tick_params(axis='both', which='major', labelsize=18)
    plt.plot(sigma2mass,sigma2xsec,'o-',alpha=1.0,markersize=5)
    plt.plot(sigma1mass,sigma1xsec,'o-',alpha=1.0,markersize=5)
    plt.plot(bestmass,bestxsec,'o',alpha=1.0,markersize=10)
    plt.xlabel(r'Mass (GeV/c$^2$)',fontsize=24)
    plt.ylabel(r'$\sigma_N$ (cm$^2$)',fontsize=24)
    plt.yscale('log')
    plt.tight_layout()

    #name = "Plots/contours_%s.png" % (tag)
    #plt.savefig(name)


    plt.show()


if __name__ == "__main__":
    plot_contours_for_scans()
