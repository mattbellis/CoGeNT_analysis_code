import numpy as np
import lichen.lichen as lch
import matplotlib.pyplot as plt

import sys

vals = np.loadtxt(sys.argv[1])

days = vals[:,0]
energies = vals[:,1]
risetimes = vals[:,2]

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
lch.hist_err(energies,bins=50)
plt.xlabel('Recoil energy (keVee)')
plt.ylim(0)

plt.subplot(1,3,2)
lch.hist_err(days,bins=50)
plt.xlabel('Days since XXX')
plt.ylim(0)

plt.subplot(1,3,3)
lch.hist_err(risetimes,bins=50,range=(0,6))
plt.xlabel(r'Rise times ($\mu$s)')
plt.ylim(0)
plt.xlim(0,6)

plt.tight_layout()


'''
# Scatter
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(energies,days,'o',markersize=1,alpha=0.1)
plt.xlabel('Recoil energy (keVee)')
plt.ylabel('Days since XXX')
#plt.ylim(0)

plt.subplot(1,3,2)
plt.plot(days,risetimes,'o',markersize=1,alpha=0.1)
plt.xlabel('Days since XXX')
plt.ylabel(r'Rise times ($\mu$s)')
#plt.ylim(0)

plt.subplot(1,3,3)
plt.plot(energies,risetimes,'o',markersize=1,alpha=0.1)
plt.xlabel('Recoil energy (keVee)')
plt.ylabel(r'Rise times ($\mu$s)')
#plt.ylim(0)
#plt.xlim(0,6)

plt.tight_layout()

'''

plt.show()


