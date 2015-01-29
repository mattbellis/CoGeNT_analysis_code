import numpy as np
import csv

import sys

################################################################################
################################################################################
def write_output_file(energy,time_stamps,rise_times,file_name):

    #zip(energy,time_stamps,rise_times)
    with open(file_name,'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(energy,time_stamps,rise_times))
        f.close()


################################################################################

nsamples = int(sys.argv[1])

'''
infile_names = ['MC_files/mc_surface_bulk_samples_1M',
                'MC_files/mc_flat_bulk_samples_1M',
                'MC_files/mc_lshell_bulk_samples_1M']
'''

infile_names = ['MC_files/mc_surface_bulk_samples',
                'MC_files/mc_flat_bulk_samples',
                'MC_files/mc_lshell_bulk_samples']

central_values = [4482, 3140, 900]

infiles = []
bulk_bkgs = []
bkgs = []
nentries = []
for name in infile_names:

    infile = open(name)

    bkg = np.loadtxt(infile)
    bkgs.append(bkg)

    nentries.append(len(bkg))


indices = []
for n in nentries:
    index = np.arange(0,n,1)
    indices.append(index)

for i in xrange(0,nsamples):
    tot_bkgs = np.array([])
    num_to_grab = []
    for bkg,index,cv in zip(bkgs,indices,central_values):

        np.random.shuffle(index)
        ng = np.random.poisson(cv)
        new_bkg = bkg[index][0:ng]

        num_to_grab.append(ng)

        #print new_bkg
        tot_bkgs = np.append(tot_bkgs,new_bkg)
        #print tot_bkgs

    testname = "MC_files/sample_surf_%d_%d_flat_%d_%d_lshell_%d_%d_%04d.dat" % (central_values[0],num_to_grab[0],central_values[1],num_to_grab[1],central_values[2],num_to_grab[2],i)
    print testname
    #print tot_bkgs.shape
    #print len(tot_bkgs)
    index = np.arange(0,len(tot_bkgs),3)
    #print tot_bkgs
    write_output_file(tot_bkgs[index],tot_bkgs[index+1],tot_bkgs[index+2],testname)


