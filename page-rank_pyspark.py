import sys
import heapq
import numpy as np
import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.mllib.linalg import DenseMatrix
from pyspark.mllib.linalg.distributed import RowMatrix

def make_key_value_pair_1(curr_line):
	if(curr_line is not None):
		source = int(curr_line.split()[0])
		destination = int(curr_line.split()[1])
		return source, destination

def make_key_value_pair_2(curr_line):
	if(curr_line is not None):
		source = int(curr_line.split()[0])
		destination = int(curr_line.split()[1])
		return (source, destination), 1

def calc_out_degree(curr_pair):
	if curr_pair is not None:
		return curr_pair[0], len(list(curr_pair[1]))

def main():
	datasetfile = sys.argv[1]
	beta = 0.8
	iterations = 40
	top_k = 5

	sparkcontext = SparkContext("local", "Page Rank")
	data = sparkcontext.textFile(datasetfile)
	source_dest = data.map(make_key_value_pair_1)
	source_dest_count = data.map(make_key_value_pair_2)
	groupbykey = source_dest.groupByKey()
	number_of_nodes = groupbykey.count()
	out_degree = groupbykey.map(calc_out_degree)
	pair_map = groupbykey.collectAsMap()

	matrix_m = np.zeros(shape=(number_of_nodes,number_of_nodes))
	for key,value in pair_map.items():
		for ind_value in value:
			matrix_m[ind_value-1][key-1] += 1/len(list(value))

	matrix_m = sparkcontext.parallelize(matrix_m)
	matrix_m = RowMatrix(matrix_m)

	vector_r_prev = np.empty([number_of_nodes,1])
	vector_r_prev.fill(1/number_of_nodes)
	vector_r_prev = DenseMatrix(number_of_nodes,1,vector_r_prev)

	index = 0
	while(index < iterations):
		mul_val = matrix_m.multiply(vector_r_prev).rows.collect()
		mul_val = [i*beta for i in mul_val]
		mul_val = [i+(1-beta)/number_of_nodes for i in mul_val]
		vector_r_prev = DenseMatrix(number_of_nodes,1,mul_val)
		index += 1

	vector_r_prev = vector_r_prev.toArray()
	largest_values = heapq.nlargest(top_k, vector_r_prev)
	largest_indexes = heapq.nlargest(top_k, range(number_of_nodes), vector_r_prev.__getitem__)
	smallest_values = heapq.nsmallest(top_k, vector_r_prev)
	smallest_indexes = heapq.nsmallest(top_k, range(number_of_nodes), vector_r_prev.__getitem__)

	largest_indexes = [val+1 for val in largest_indexes]
	smallest_indexes = [val+1 for val in smallest_indexes]

	print("Value of largest n nodes\n", largest_values)
	print("Node numbers of largest n nodes\n", largest_indexes)
	print("Value of smallest n nodes\n", smallest_values)
	print("Node numbers of smallest n nodes\n", smallest_indexes)
	sparkcontext.stop()

if __name__ == '__main__':
	main()
