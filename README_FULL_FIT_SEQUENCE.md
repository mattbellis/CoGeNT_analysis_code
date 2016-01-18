# Get the rise-time parameters
# First from Nicole

python fit_rise_times_pulser.py --dataset nicole --batch
python fit_rise_times_of_data_using_pulser_fit_inputs.py --rt-parameters risetime_parameters_risetime_determination_nicole.py --batch


# Then from Juan

python fit_rise_times_pulser.py --dataset juan --batch
python fit_rise_times_of_data_using_pulser_fit_inputs.py --rt-parameters risetime_parameters_risetime_determination_juan.py --batch


# Then try a fit. 
python fit_cogent_data.py --rt-parameters risetime_parameters_from_data_risetime_parameters_risetime_determination_nicole.py --batch

# Can also compare to Juans data.
python fit_cogent_data.py --rt-parameters risetime_parameters_from_data_risetime_parameters_risetime_determination_juan.py --batch
