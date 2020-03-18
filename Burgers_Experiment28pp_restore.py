import pickle
import copy
import glob, os
import math
import training

def BestErrorFile(folder_name):
	
	filename_best = ''
	best_error = 10**6

	current_dir = os.getcwd()
	os.chdir(folder_name)
	for file in glob.glob("*.pkl"):
		with open(file, 'rb') as f:
			params = pickle.load(f)
			if not math.isnan(params['minRegTest']):
				if params['minRegTest'] < best_error:
					filename_best = file
					best_error = params['minRegTest']
        os.chdir(current_dir)

	return filename_best

folder_name = './Burgers_exp28pp/'
file_name = BestErrorFile(folder_name)
if file_name == '':
	print('No files have finite error')
	quit()
else:
	pkl_file = folder_name + file_name

with open(pkl_file, 'rb') as f:
     params = pickle.load(f)

print(params['minRegTest'])

params['restore'] = 1
params['auto_first'] = 0
params['max_time'] = 12*60*60
params['min_5min'] = 10**2
params['min_20min'] = 10**1
params['min_40min'] = 5
params['min_1hr'] = 3
params['min_2hr'] = 2
params['min_3hr'] = 2
params['min_4hr'] = 2
params['min_halfway'] = 2

params['model_restore_path'] = params['model_path']

training.main_exp(copy.deepcopy(params))
