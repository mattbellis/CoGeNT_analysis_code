import numpy as np
import csv
from cogent_utilities import cogent_efficiency

import sys

import parameters

################################################################################
################################################################################
def write_output_file(time_stamps,energy,rise_times,file_name):

    #zip(energy,time_stamps,rise_times)
    with open(file_name,'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(time_stamps,energy,rise_times))
        f.close()


################################################################################
def cut_subranges(data,subrange,which_one=0):

    if len(subrange)==0:
        return data

    #print data
    #print data.shape
    re_data = data.transpose()
    selection_variable = re_data[which_one]

    #print selection_variable

    index = np.zeros(len(data),dtype=int)
    for r in subrange:
        #print r
        #print selection_variable
        index_lo = selection_variable>r[0]
        index_hi = selection_variable<=r[1]

        index += index_lo*index_hi

        #print len(index[index>0])

    #print len(index[index>0])
    #print index
    new_data = data[index.astype(bool)].copy()
    #print len(new_data)

    return new_data



################################################################################

nsamples = 1
if len(sys.argv)>1:
    nsamples = int(sys.argv[1])

bkg_names = ['surface','neutron','compton','lshell','shm_wimps']
tag = '100k'
#tag = '1M'
#tag = '10k'
infile_names = []
for b in bkg_names:
    name = "MC_files/mc_%s_bulk_samples_%s.dat" % (b,tag)
    infile_names.append(name)

#central_values = [4482, 3140, 975]
#central_values = [4482, 862, 2287, 975]
#central_values = [7920, 1440, 4460, 1400, 0] # Background only for FIT0001
#central_values = [7920, 1440, 4460, 1400, 385] # Adding WIMP for FIT0002, #WIMPS=385 for 0.5-4.0
central_values = [7920, 1440, 4460, 0, 0] # Background only test, FIT003, no l-shell
#central_values = [7000, 1500, 0, 1250]
#central_values = [0, 1500, 1500, 0]
#central_values = [1000, 1500, 0, 0, 385]
#central_values = [1000, 1500, 1500, 0, 0]
ranges,subranges,nbins = parameters.fitting_parameters(0)

#data = [tdays,energies,rise_time]

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

    infile = open(name)
    bkg = np.loadtxt(infile)
    nentries.append(len(bkg))

    bkgs.append(bkg)


indices = []
for n in nentries:
    index = np.arange(0,n,1)
    indices.append(index)

for i in xrange(0,nsamples):
    tot_bkgs = np.array([])
    num_to_grab = []
    #testname = "MC_files/sample"
    #testname = "MC_files/sample_FIT0001"
    #testname = "MC_files/sample_FIT0002"
    testname = "MC_files/sample_FIT0003"
    for bkg,bkg_name,index,cv in zip(bkgs,bkg_names,indices,central_values):

        np.random.shuffle(index)
        ng = np.random.poisson(cv)
        new_bkg = bkg[index][0:ng]

        # Original number
        testname += "_%s_%d" % (bkg_name,ng)

        #print new_bkg.shape

        # Apply the CoGeNT efficiency.
        days = new_bkg[:,0]
        energies = new_bkg[:,1]
        risetimes = new_bkg[:,2]
        data = [days,energies,risetimes]
        data = cogent_efficiency(data,threshold,sigmoid_sigma,max_val)
        #print type(data)
        #print new_bkg
        new_bkg = np.array([data[0],data[1],data[2]])
        new_bkg = new_bkg.transpose()
        #print new_bkg.shape

        testname += "_%d" % (len(new_bkg))

        # Cut out the deadtime regions.
        cut_bkg = cut_subranges(new_bkg,subranges[1],0)

        testname += "_%d" % (len(cut_bkg))

        #print "cut len: %d" % (len(cut_bkg))

        #num_to_grab.append(ng)
        #num_to_grab.append(len(cut_bkg))

        #print cut_bkg
        tot_bkgs = np.append(tot_bkgs,cut_bkg)
        #print tot_bkgs

    #testname = "MC_files/sample_surf_%d_%d_neutrons_%d_%d_comptons_lshell_%d_%d_%04d.dat" % (central_values[0],num_to_grab[0],central_values[1],num_to_grab[1],central_values[2],num_to_grab[2],i)
    testname += "_from_%s_samples_%03d.dat" % (tag,i)
    print testname
    #print tot_bkgs.shape
    #print len(tot_bkgs)
    index = np.arange(0,len(tot_bkgs),3)
    #print tot_bkgs
    write_output_file(tot_bkgs[index],tot_bkgs[index+1],tot_bkgs[index+2],testname)


