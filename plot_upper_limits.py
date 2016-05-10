import matplotlib,pylab as plt
import seaborn as sn

import numpy as np

#filenames = ['upper_limits_0.50-3.2_scans_juan.dat',
             #'upper_limits_0.55-3.2_scans_juan.dat',
            #'upper_limits_0.50-3.2_scans_nicole.dat',
            #'upper_limits_0.55-3.2_scans_nicole.dat'
        #]

filenames = ['upper_limits_scans_juan_stream_0.50-3.2.dat',
             'upper_limits_scans_juan_stream_0.55-3.2.dat',
             'upper_limits_scans_nicole_stream_0.50-3.2.dat',
             'upper_limits_scans_nicole_stream_0.55-3.2.dat'
        ]

labels = [r'Surf. events param. #1 (E$_{\rm low}$=0.50 keVee)',
          r'Surf. events param. #1 (E$_{\rm low}$=0.55 keVee)',
          r'Surf. events param. #2 (E$_{\rm low}$=0.50 keVee)',
          r'Surf. events param. #2 (E$_{\rm low}$=0.55 keVee)']

files = []

fig = plt.figure(figsize=(10,5))
for i,fn in enumerate(filenames):
    #files.append(open(fn))

    x,y = np.loadtxt(fn,unpack=True,dtype=float)

    #x = x.astype(float)
    #y = y.astype(float)

    plt.plot(x,y,'-',label=labels[i],linewidth=4,alpha=0.80)


plt.yscale('log')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel(r'WIMP-nucleon $\sigma$ [cm$^2$]',fontsize=24)
plt.xlabel(r'WIMP mass [GeV/c$^2$]',fontsize=24)
plt.legend(loc='upper right',fontsize=18)
plt.tight_layout()

plt.savefig('upper_limits.png')

plt.show()


