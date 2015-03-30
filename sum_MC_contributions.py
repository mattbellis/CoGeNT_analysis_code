import numpy as np
import lichen.lichen as lch
import matplotlib.pyplot as plt

from cogent_utilities import cogent_efficiency,cut_events_outside_subrange,cut_events_outside_range

import sys

import parameters

bkg_names = ['surface','neutron','compton','lshell','shm_wimps']
bkg_colors = ['y','c','m','r','k','b']
#tag = '100k'
#tag = '1M'
tag = '10k'
infile_names = []
for b in bkg_names:
    name = "MC_files/mc_%s_bulk_samples_%s.dat" % (b,tag)
    if b=='shm_wimps':
        name = "MC_files/mc_%s_bulk_samples_%s.dat" % (b,'10k')
    infile_names.append(name)

if len(sys.argv)>1:
    infile_names.append(sys.argv[1])
    bkg_names.append('data')

#bkg_names = ['surface','neutron','compton','lshell','shm_wimps']
#central_values = [4580.5773, 1522.5595, 1530.0256, 983.3246, 0,0]
central_values = [715,1094,0,0,0,0]
#central_values = [581, 865, 0, 0, 265,0]
ranges,subranges,nbins = parameters.fitting_parameters(0)


# CoGeNT efficiency parameters
threshold = 0.345
sigmoid_sigma = 0.241
max_val = 0.86786

infiles = []
bulk_bkgs = []
bkgs = []
nentries = []
# Read in the data.
for name in infile_names:
    print "Reading in %s" % (name)
    infile = open(name)
    bkg = np.loadtxt(infile)
    nentries.append(len(bkg))
    bkgs.append(bkg)

# Apply efficiency and cuts to data
fig = plt.figure(figsize=(12,6))

new_bkgs = []
for i,(bkg,bkg_name,cv) in enumerate(zip(bkgs,bkg_names,central_values)):

    # Apply the CoGeNT efficiency.
    days = bkg[:,0]
    energies = bkg[:,1]
    risetimes = bkg[:,2]
    data = [energies,days,risetimes]
    data = cogent_efficiency(data,threshold,sigmoid_sigma,max_val)

    #new_bkg = np.array([data[1],data[0],data[2]])
    new_bkg = [data[0],data[1],data[2]]
    #new_bkg = new_bkg.transpose()

    print new_bkg

    # Cut out dead-time ranges.
    cut_ranges_bkg = cut_events_outside_range(new_bkg,ranges)
    print len(cut_ranges_bkg[0])
    cut_bkg = cut_events_outside_subrange(cut_ranges_bkg,subranges[1],data_index=1)
    print len(cut_bkg[0])

    #cut_bkg = np.array(cut_bkg)
    #cut_bkg.transpose()

    print cut_bkg

    new_bkgs.append(cut_bkg)

    for j in xrange(3):
        plt.subplot(1,3,j+1)
        #print cv
        #print data[j][0:10]
        if cv>0:
            lch.hist_err(data[j],color=bkg_colors[i],ecolor=bkg_colors[i])

plt.tight_layout()

print "HERE A"

fig2 = plt.figure(figsize=(12,6))
for j in xrange(3):
    plt.subplot(1,3,j+1)
    data = []
    colors = []
    weights = []
    for k in range(0,5):
        print new_bkgs[k]
        if len(new_bkgs[k][j])>0:
            data.append(new_bkgs[k][j])
            colors.append(bkg_colors[k])
            bin_width=(ranges[j][1]-ranges[j][0])/nbins[j]
            #print bin_width
            nentries = float(len(new_bkgs[k][j]))
            print "nentries: ",nentries
            weights.append((central_values[k]/nentries)*np.ones(nentries))
            #weights.append(np.ones(nentries))
        #print new_bkgs[k][j]
    #print "WEIGHTS"
    #print weights
    #print data
    plt.hist(data,stacked=True,bins=nbins[j],range=(ranges[j][0],ranges[j][1]),histtype='stepfilled',color=colors,weights=weights)

    if len(sys.argv)>1:
        #print new_bkgs[5]
        lch.hist_err(new_bkgs[5][j],bins=nbins[j],range=(ranges[j][0],ranges[j][1]),markersize=5,linewidth=1,ecolor='blue')

plt.tight_layout()

plt.show()


