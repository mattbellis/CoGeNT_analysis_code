import matplotlib.pylab as plt
import numpy as np
import chris_kelso_code as dmm
from chris_kelso_code import dRdErSHM
import csv
import numpy as np
from cogent_utilities import *
from cogent_pdfs import surface_events,flat_events,compton_events,neutron_events
# This is just for the Siena HPC cluster
#import sys
#sys.path.append("/home/mbellis/python/lib/python/")
#sys.path.append("/home/mbellis/python/lib64/python/")
import lichen.pdfs as pdfs
import lichen.lichen as lch

import datetime

import parameters

import sys

################################################################################
# Slow parameters
# Using Nicole's simulated stuff
mu = [0.269108,0.747275,0.068146]
sigma = [0.531530,-0.020523]

# Fast parameters
#Using Nicole's simulated stuff
fast_mean_rel_k = [0.431998,-1.525604,-0.024960]
fast_sigma_rel_k = [-0.014644,5.745791,-6.168695]
fast_num_rel_k = [-0.261322,5.553102,-5.9144]

mu0 = [0.374145,0.628990,-1.369876]
sigma0 = [1.383249,0.495044,0.263360]
################################################################################

################################################################################
################################################################################
def write_output_file(energy,time_stamps,rise_times,file_name):

    #zip(energy,time_stamps,rise_times)
    with open(file_name,'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(list(zip(time_stamps,energy,rise_times)))
        f.close()

################################################################################
################################################################################
def gen_compton_events(maxpts,max_days,name_of_output_file,pars):

    ranges,subranges,nbins = parameters.fitting_parameters(0)

    lo = [ranges[0][0],ranges[1][0]]
    hi = [ranges[0][1],ranges[1][1]]

    elo = lo[0]
    ehi = hi[0]

    #max_prob = 3.3
    max_prob = 0.00155
    print(("Max prob currently is: %f" % (max_prob)))
    energies = []
    days = []
    rise_times = []

    npts = 0
    max_prob_calculated = -999
    while npts < maxpts:

        e = ((ehi-elo)*np.random.random(1) + elo) # This is the energy
        t = (max_days)*np.random.random(1)
        rt = (6.0)*np.random.random(1)

        rt_fast = rise_time_prob_fast_exp_dist(rt,e,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])

        data = [e,t,0,rt_fast,0]

        prob = compton_events(data,pars['final_values'],lo,hi,subranges=subranges,efficiency=None)

        if max_prob_calculated<prob:
            print(("Max prob to now: %f" % (prob)))
            max_prob_calculated = prob
            max_prob = prob

        '''
        if max_prob<prob:
            print prob
        '''

        probtest = max_prob*np.random.random() # This is to see whether or not we keep x!

        if probtest<prob:
            energies.append(e[0])
            days.append(t[0])
            rise_times.append(rt[0])
            npts += 1
            if npts%1000==0:
                print(npts)


    write_output_file(energies,days,rise_times,name_of_output_file)
                            
    return energies,days,rise_times

################################################################################
################################################################################
def gen_neutron_events(maxpts,max_days,name_of_output_file,pars):

    ranges,subranges,nbins = parameters.fitting_parameters(0)

    lo = [ranges[0][0],ranges[1][0]]
    hi = [ranges[0][1],ranges[1][1]]

    elo = lo[0]
    ehi = hi[0]

    max_prob = 0.00096
    #max_prob = 0.0012
    #max_prob = 3.3
    print(("Max prob currently is: %f" % (max_prob)))
    energies = []
    days = []
    rise_times = []

    npts = 0
    max_prob_calculated = -999
    while npts < maxpts:

        e = ((ehi-elo)*np.random.random(1) + elo) # This is the energy
        t = (max_days)*np.random.random(1)
        rt = (6.0)*np.random.random(1)

        rt_fast = rise_time_prob_fast_exp_dist(rt,e,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])

        data = [e,t,0,rt_fast,0]

        prob = neutron_events(data,pars['final_values'],lo,hi,subranges=subranges,efficiency=None)

        if max_prob_calculated<prob:
            print(("Max prob to now: %f" % (prob)))
            max_prob_calculated = prob
            max_prob = prob

        '''
        if max_prob<prob:
            print prob
        '''

        probtest = max_prob*np.random.random() # This is to see whether or not we keep x!

        if probtest<prob:
            energies.append(e[0])
            days.append(t[0])
            rise_times.append(rt[0])
            npts += 1
            if npts%1000==0:
                print(npts)


    write_output_file(energies,days,rise_times,name_of_output_file)
                            
    return energies,days,rise_times

################################################################################
################################################################################
def gen_flat_events(maxpts,max_days,name_of_output_file,pars):

    #ranges,subranges,nbins = parameters.fitting_parameters(0)
    ranges = [[8.0,12.0],[1.0,max_days],[0.0,6.0]]
    subranges = [[],[[1,68],[75,102],[108,306],[309,459],[551,1238]],[]]

    lo = [ranges[0][0],ranges[1][0]]
    hi = [ranges[0][1],ranges[1][1]]

    elo = lo[0]
    ehi = hi[0]

    max_prob = 4.533
    print(("Max prob currently is: %f" % (max_prob)))
    energies = []
    days = []
    rise_times = []

    npts = 0
    max_prob_calculated = -999
    while npts < maxpts:

        e = ((ehi-elo)*np.random.random(1) + elo) # This is the energy
        t = (max_days)*np.random.random(1)
        rt = (6.0)*np.random.random(1)

        rt_fast = rise_time_prob_fast_exp_dist(rt,e,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])

        data = [e,t,0,rt_fast,0]

        prob = flat_events(data,pars,lo,hi,subranges=subranges,efficiency=None)

        if max_prob_calculated<prob:
            print(("Max prob to now: %f" % (prob)))
            max_prob_calculated = prob
            max_prob = prob

        '''
        if max_prob<prob:
            print prob
        '''

        probtest = max_prob*np.random.random() # This is to see whether or not we keep x!

        if probtest<prob:
            energies.append(e[0])
            days.append(t[0])
            rise_times.append(rt[0])
            npts += 1
            if npts%1000==0:
                print(npts)


    write_output_file(energies,days,rise_times,name_of_output_file)
                            
    return energies,days,rise_times

################################################################################
################################################################################
def gen_peaks(maxpts,max_days,name_of_output_file,means=[10],sigmas=[0.11],numls=[200],decay_constants=[0.000000001]):#,pars):

    #ranges,subranges,nbins = parameters.fitting_parameters(0)
    #print(ranges)
    #print(subranges)

    # Energy, days, rise time
    #ranges = [ [9.0, 11.0], [0, max_days], [0.0, 6.0] ]
    ranges = [[8.0,12.0],[1.0,max_days],[0.0,6.0]]
    subranges = [[],[[1,68],[75,102],[108,306],[309,459],[551,1238]],[]]

    lo = [ranges[0][0],ranges[1][0]]
    hi = [ranges[0][1],ranges[1][1]]


    ############################################################################
    # l-shell peaks
    ############################################################################
    #means = [10]
    #sigmas = [0.11]
    #numls = [200]
    #decay_constants = [0.00000001]
    
    num_tot = 0

    npts_to_generate = 100

    '''
    for i in range(11):
        name = "ls_mean%d" % (i)
        means.append(pars['final_values'][name])
        name = "ls_sigma%d" % (i)
        sigmas.append(pars['final_values'][name])
        name = "ls_ncalc%d" % (i)
        #numls.append(pars['final_values'][name]/num_tot) # Normalized this # to number of events.
        numls.append(pars['final_values'][name])
        num_tot += pars['final_values'][name]
        name = "ls_dc%d" % (i)
        decay_constants.append(pars['final_values'][name])
    '''

    #max_prob = 26.0
    max_prob = 26.0
    print(("Max prob currently is: %f" % (max_prob)))
    energies = np.zeros(maxpts)
    days = np.zeros(maxpts)
    rise_times = np.zeros(maxpts)

    npts = 0
    max_prob_calculated = -999
    efficiency = None
    while npts < maxpts:

        e = ((hi[0]-lo[0])*np.random.random(npts_to_generate) + lo[0]) # Generate over a smaller range for the L-shell peaks
        t = (max_days)*np.random.random(npts_to_generate)
        rt = (3.0)*np.random.random(npts_to_generate)

        rt_fast = rise_time_prob_fast_exp_dist(rt,e,mu0,sigma0,fast_mean_rel_k,fast_sigma_rel_k,fast_num_rel_k,ranges[2][0],ranges[2][1])

        data = [e,t,0,rt_fast,0]

        tot_pdf = 0
        for n,m,s,dc in zip(numls,means,sigmas,decay_constants):
            #print(n,m,s,dc,lo[0],hi[0])
            pdf  = pdfs.gauss(e,m,s,lo[0],hi[0],efficiency=efficiency)
            #print(e[0:10],t[0:10],pdf[0:10])
            dc = -1.0*dc
            pdf *= pdfs.exp(t,dc,lo[1],hi[1],subranges=subranges[1])
            pdf *= rt_fast # This will be the fast rise times
            pdf *= n
            tot_pdf += pdf

        prob = tot_pdf
        
        #print "MAX:"
        #print max(prob)
        if max(prob)>max_prob_calculated:
            print(("Max prob to now: %f" % max(prob)))
            max_prob_calculated = max(prob)
            max_prob = max(prob)

        if len(prob[prob>max_prob])>0:
            print((max_prob,prob[prob>max_prob]))

        probtest = max_prob*np.random.random(npts_to_generate) # This is to see whether or not we keep x!

        #if probtest<prob:
        index = probtest<prob
        npts_good = len(index[index==True])
        if len(index)>0:
            #print "-----------"
            #print energies[npts:npts+npts_good] 
            #print e[index]

            final_index = npts+npts_good
            max_to_insert = npts_good
            #print len(energies),final_index,npts
            if final_index>len(energies):
                final_index = len(energies)
                max_to_insert = len(energies)-npts

            #print "final_index: ",final_index
            energies[npts:final_index] = e[index][0:max_to_insert]
            days[npts:final_index] = t[index][0:max_to_insert]
            rise_times[npts:final_index] = rt[index][0:max_to_insert]
            
            npts += npts_good

            #print(npts)
            #if npts%1000==0:
            #print npts

            #print npts

    write_output_file(energies,days,rise_times,name_of_output_file)
                            
    return energies,days,rise_times


################################################################################
# Read in from a previous fits of results.
#results_file = open(sys.argv[1])
results_file = open('results_bkg_only_for_MC_gen.txt')
results = eval(results_file.readline())
#print results
#exit()


print("Generating data!!!!!")
#print datetime.datetime.now()

nevents = 200
ndays = 5*365

tag = "test_data"
#tag = "bulk_samples_%ddays_%devents" % (ndays,nevents)


etot = np.array([])
dtot = np.array([])
rtot = np.array([])

print("Generating peaks......")
print((datetime.datetime.now()))
name = "MC_files/peaks_%s.dat" % (tag)
#energies,days,rise_times = gen_peaks(200,ndays,name,means=[10],sigmas=[0.11],numls=[200],decay_constants=[0.000000001])# ,results)
energies,days,rise_times = gen_peaks(1000,ndays,name,means=[9.8,10.1],sigmas=[0.11,0.11],numls=[100,40],decay_constants=[1e-6,1e-6])# ,results)
etot = np.append(etot,energies)
dtot = np.append(dtot,days)
rtot = np.append(rtot,rise_times)
print((datetime.datetime.now()))

#################################
print("Generating flat......")
print((datetime.datetime.now()))
name = "MC_files/flat_%s.dat" % (tag)
energies,days,rise_times = gen_flat_events(1000,ndays,name,results)
etot = np.append(etot,energies)
dtot = np.append(dtot,days)
rtot = np.append(rtot,rise_times)
print((datetime.datetime.now()))

name = "MC_files/%s.dat" % (tag)
write_output_file(etot,dtot,rtot,name)


#'''
nbins = 50
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
h = plt.hist(etot,bins=nbins)
plt.subplot(1,3,2)
h = plt.hist(dtot,bins=nbins)
plt.subplot(1,3,3)
h = plt.hist(rtot,bins=nbins)

plt.show()
#'''
'''
nbins = 50
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
h = lch.hist_err(etot,bins=nbins)
plt.xlabel('Energy (keVee)')

plt.subplot(1,3,2)
h = lch.hist_err(dtot,bins=nbins)
plt.xlabel('Time (days)')

plt.subplot(1,3,3)
h = lch.hist_err(rtot,bins=nbins)
plt.xlabel('Rise time ($\mu$s)')

plt.tight_layout()
'''


#plt.show()
