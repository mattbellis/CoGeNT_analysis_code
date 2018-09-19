import numpy as np
import matplotlib.pylab as plt

import seaborn as sns
sns.set('poster')




import cogent_utilities as cog

import sys

first_event = 2750361.2

infilename = sys.argv[1]

tdays,energies,rise_time = cog.get_3yr_cogent_data(infilename,first_event=first_event,calibration=0)

print(energies)

hrange = (0.5,12)
bins = 300

plt.figure(figsize=(15,5))
plt.hist(energies, bins=bins, range=hrange, label='Full 3.5 years')

plt.hist(energies[tdays>365], bins=bins, range=hrange, color='orange', label='After first year of running')
plt.yscale('log')
plt.xlabel('Energy (keVee)', fontsize=18)

plt.gca().xaxis.set_major_locator(plt.MaxNLocator(13))

plt.legend(fontsize=18)

plt.tight_layout()

print(tdays)
plt.savefig('cogent_dataset.png')

plt.show()
