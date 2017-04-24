#!/usr/bin/env python3

import hdim
import test_data_gen
import numpy as np
import matplotlib.pyplot as plt
import math

def test_FOS( X, Y ):
	fos_test = hdim.FOS_d( X, Y )

	fos_test.Algorithm()
	return fos_test.ReturnSupport()

def test_X_FOS( X, Y ):
	fos_test = hdim.X_FOS_d()

	fos_test( X, Y )
	return fos_test.ReturnSupport()

def main():
	N = 50
	P = 100

	#ind = np.arange(P)  # the x locations for the groups

	X, y, beta = test_data_gen.generate_data( N, P, int( math.ceil(N/10) ), 0.3, 5 )

	fos_results= test_FOS( X, y )
	x_fos_results = test_X_FOS( X, y )

	ind = np.arange( len( fos_results ) )

	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, fos_results, width, color='#800080')
	rects2 = ax.bar(ind + width, x_fos_results, width, color='y')

	ax.legend((rects1[0], rects2[0]), ('FOS', 'X_FOS'))

	plt.show()

if __name__ == "__main__":
    main()
