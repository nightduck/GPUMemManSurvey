import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('../../scripts')

import os
import shutil
import time
from datetime import datetime
from timedprocess import Command
import pandas
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

colours = {
	'Halloc' : 'orange' , 
	'Ouroboros-P-VA' : 'lightcoral' , 'Ouroboros-P-VL' : 'darkred' , 'Ouroboros-P-S' : 'red' ,
	'Ouroboros-C-VA' : 'red' , 'Ouroboros-C-VL' : 'red' , 'Ouroboros-C-S' : 'red' ,
	'CUDA' : 'green' , 
	'ScatterAlloc' : 'blue' , 
	'FDGMalloc' : 'gold' , 
	'RegEff-A' : 'mediumvioletred' , 'RegEff-AW' : 'orchid',
	'RegEff-C' : 'purple' , 'RegEff-CF' : 'violet' , 'RegEff-CM' : 'indigo' , 'RegEff-CFM' : 'blueviolet'
}

import argparse

def main():
	# Run all files from a directory
	print("##############################################################################")
	print("Callable as: python test_allocation.py")
	print("##############################################################################")
	
	# Specify which test configuration to use
	testcases = list()
	num_allocations = 10000
	smallest_allocation_size = 4
	largest_allocation_size = 1024
	num_iterations = 25
	free_memory = 1
	build_path = "build/"

	parser = argparse.ArgumentParser(description='Test allocation performance for various frameworks')
	parser.add_argument('-t', type=str, help='Specify which frameworks to test, separated by +, e.g. o+s+h+c+f+r ---> c : cuda | s : scatteralloc | h : halloc | o : ouroboros | f : fdgmalloc | r : register-efficient')
	parser.add_argument('-num', type=int, help='How many allocations to perform')
	parser.add_argument('-range', type=str, help='Sepcify Allocation Range, e.g. 4-1024')
	parser.add_argument('-iter', type=int, help='How many iterations?')
	parser.add_argument('-genres', action='store_true', default=False, help='Run testcases and generate results')
	parser.add_argument('-genplot', action='store_true', default=False, help='Generate results file and plot')
	parser.add_argument('-cleantemp', action='store_true', default=False, help='Clean up temporary files')
	parser.add_argument('-warp', action='store_true', default=False, help='Start testcases warp-based')

	args = parser.parse_args()

	# Parse approaches
	if(args.t):
		if any("h" in s for s in args.t):
			testcases.append(build_path + str("h_alloc_test"))
		if any("s" in s for s in args.t):
			testcases.append(build_path + str("s_alloc_test"))
		if any("o" in s for s in args.t):
			testcases.append(build_path + str("o_alloc_test_p"))
			testcases.append(build_path + str("o_alloc_test_c"))
			testcases.append(build_path + str("o_alloc_test_vap"))
			testcases.append(build_path + str("o_alloc_test_vac"))
			testcases.append(build_path + str("o_alloc_test_vlp"))
			testcases.append(build_path + str("o_alloc_test_vlc"))
		if any("c" in s for s in args.t):
			testcases.append(build_path + str("c_alloc_test"))
		if any("f" in s for s in args.t):
			testcases.append(build_path + str("f_alloc_test"))
		if any("r" in s for s in args.t):
			testcases.append(build_path + str("r_alloc_test_a"))
			testcases.append(build_path + str("r_alloc_test_aw"))
			testcases.append(build_path + str("r_alloc_test_c"))
			testcases.append(build_path + str("r_alloc_test_cf"))
			testcases.append(build_path + str("r_alloc_test_cm"))
			testcases.append(build_path + str("r_alloc_test_cfm"))
	
	# Parse num allocation
	if(args.num):
		num_allocations = args.num

	# Parse range
	if(args.range):
		selected_range = args.range.split('-')
		smallest_allocation_size = int(selected_range[0])
		largest_allocation_size = int(selected_range[1])

	# Parse num iterations
	if(args.iter):
		num_iterations = args.iter

	# Generate results
	if args.warp:
		test_warp_based = 1
	else:
		test_warp_based = 0
	
	# Generate results
	generate_results = args.genres

	# Generate plots
	generate_plots = args.genplot

	# Generate plots
	clean_temporary_files = args.cleantemp

	testcases.sort()

	# Timeout (in seconds)
	time_out_val = 10

	####################################################################################################
	####################################################################################################
	# Generate new Results
	####################################################################################################
	####################################################################################################
	if generate_results:
		for executable in testcases:
			smallest_allocation_size = 4
			while smallest_allocation_size <= largest_allocation_size:
				run_config = str(num_allocations) + " " + str(smallest_allocation_size) + " " + str(num_iterations) + " " + str(test_warp_based) + " 1 " + str(free_memory) + " results/tmp/"
				executecommand = "{0} {1}".format(executable, run_config)
				print(executecommand)
				Command(executecommand).run(timeout=time_out_val)
				smallest_allocation_size += 4
	
	####################################################################################################
	####################################################################################################
	# Generate new plots
	####################################################################################################
	####################################################################################################
	if generate_plots:
		approach_result_alloc = list(list())
		approach_result_alloc.append(np.arange(smallest_allocation_size, largest_allocation_size, 4).tolist())
		approach_result_alloc[0].insert(0, "Bytes")
		approach_result_free = list(list())
		approach_result_free.append(np.arange(smallest_allocation_size, largest_allocation_size, 4).tolist())
		approach_result_free[0].insert(0, "Bytes")

		# Go over files, read data and generate new 
		for file in os.listdir("results/tmp"):
			filename = str("results/tmp/") + os.fsdecode(file)
			if(os.path.isdir(filename)):
				continue
			with open(filename, newline='') as csv_file:
				dataframe = pandas.read_csv(csv_file)
				if "free" in filename:
					approach_result_free.append(list(dataframe.iloc[:, 1]))
					approach_result_free[-1].insert(0, os.fsdecode(file).split('_')[2])
				else:
					approach_result_alloc.append(list(dataframe.iloc[:, 1]))
					approach_result_alloc[-1].insert(0, os.fsdecode(file).split('_')[2])

		# Get Timestring
		now = datetime.now()
		time_string = now.strftime("%b-%d-%Y_%H-%M-%S")

		# Generate output file
		alloc_name = str("results/") + time_string + str("_alloc_") + str(num_allocations) + str(".csv")
		with(open(alloc_name, "w")) as f:
			writer = csv.writer(f, delimiter=',')
			for row in approach_result_alloc:
				writer.writerow(row)
		
		free_name = str("results/") + time_string + str("_free_") + str(num_allocations) + str(".csv")
		with(open(free_name, "w")) as f:
			writer = csv.writer(f, delimiter=',')
			for row in approach_result_free:
				writer.writerow(row)

		# Generate output plots
		# Plot allocation plot
		df = pandas.DataFrame({str(approach_result_alloc[0][0]) : approach_result_alloc[0][1:]})
		for i in range(1, len(approach_result_alloc)):
			df[str(approach_result_alloc[i][0])] = approach_result_alloc[i][1:]
		
		# for i in range(1, len(approach_result_alloc)):
		# 	plt.plot(str(approach_result_alloc[0][0]), str(approach_result_alloc[i][0]), data=df, marker='', color=colours[str(approach_result_alloc[i][0])], linewidth=1, label=str(approach_result_alloc[i][0]))
		for i in range(1, len(approach_result_alloc)):
			sns.plot(str(approach_result_alloc[0][0]), str(approach_result_alloc[i][0]), data=df, marker='', color=colours[str(approach_result_alloc[i][0])], linewidth=1, label=str(approach_result_alloc[i][0]))
		plt.yscale("log")
		plt.ylabel('ms')
		plt.xlabel('Bytes')
		plt.title("Allocation performance for " + str(num_allocations) + " allocations")
		plt.legend()
		plt.savefig(str("results/") + time_string + "_alloc.pdf", dpi=600)

		# Clear Figure
		plt.clf()

		# Plot free plot
		df = pandas.DataFrame({str(approach_result_free[0][0]) : approach_result_free[0][1:]})
		for i in range(1, len(approach_result_free)):
			df[str(approach_result_free[i][0])] = approach_result_free[i][1:]
		
		for i in range(1, len(approach_result_free)):
			plt.plot(str(approach_result_free[0][0]), str(approach_result_free[i][0]), data=df, marker='', color=colours[str(approach_result_free[i][0])], linewidth=1, label=str(approach_result_free[i][0]))
		plt.yscale("log")
		plt.ylabel('ms')
		plt.xlabel('Bytes')
		plt.title("Free performance for " + str(num_allocations) + " allocations")
		plt.legend()
		plt.savefig(str("results/") + time_string + "_free.pdf", dpi=600)

	####################################################################################################
	####################################################################################################
	# Clean temporary files
	####################################################################################################
	####################################################################################################
	if clean_temporary_files:
		for file in os.listdir("results/tmp"):
			filename = str("results/tmp/") + os.fsdecode(file)
			if(os.path.isdir(filename)):
				continue
			os.remove(filename)

	print("Done")

if __name__ == "__main__":
	main()