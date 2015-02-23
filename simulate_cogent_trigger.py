from cogent_utilities import *

import csv
import sys

################################################################################
def write_output_file(time_stamps,energy,rise_times,file_name):

    #zip(energy,time_stamps,rise_times)
    with open(file_name,'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(time_stamps,energy,rise_times))
        f.close()


################################################################################

infile_name = sys.argv[1]

tdays,energies,rise_time = get_3yr_cogent_data(infile_name,first_event=first_event,calibration=999)

data = [tdays,energies,rise_time]

threshold = 0.345
sigmoid_sigma = 0.241
max_val = 0.86786

print len(data[0])

data = cogent_efficiency(data,threshold,sigmoid_sigma,max_val)

print len(data[0])

testname = "efficiency_applied.dat"
write_output_file(data[0],data[1],data[2],testname)


