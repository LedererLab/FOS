#!/usr/bin/env python3

import hdim
import test_data_gen
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
import time

def test_X_FOS( X, Y, solver_type ):
	fos_test = hdim.X_FOS_d()

	fos_test( X, Y, solver_type )
	return fos_test.ReturnCoefficients()

def main():
	N = 200
	P = 500

	ind = np.arange(P)  # the x locations for the groups

	X, y, beta = test_data_gen.generate_data( N, P, int( math.ceil(P/10) ), 0.3, 5 )

	fos_ista_results = test_X_FOS( X, y, hdim.SolverType_ista )

	start = time.clock()
	fos_fista_results = test_X_FOS( X, y, hdim.SolverType_fista )
	end = time.clock()

	print( "FISTA timing ", end - start )

	start = time.clock()
	fos_fista_screen_results = test_X_FOS( X, y, hdim.SolverType_screen_fista )
	end = time.clock()

	print( "FISTA w/ Screening Rules timing ", end - start )

	start = time.clock()
	fos_cd_results = test_X_FOS( X, y, hdim.SolverType_cd )
	end = time.clock()

	print( "CD timing ", end - start )

	start = time.clock()
	fos_screen_cd_results = test_X_FOS( X, y, hdim.SolverType_screen_cd )
	end = time.clock()

	print( "CD w/ Screening Rules timing ", end - start )

	ind = np.arange( len( fos_ista_results ) )

	width = 0.2       # the width of the bars

	fig, ax = plt.subplots()

	rects1 = ax.bar(ind, fos_ista_results, width, color='#800080')
	rects2 = ax.bar(ind + width, fos_fista_results, width, color='b')
	rects3 = ax.bar(ind + 2*width, fos_fista_screen_results, width, color='g')
	rects4 = ax.bar(ind + 3*width, fos_cd_results, width, color='y')
	rects5 = ax.bar(ind + 4*width, fos_screen_cd_results, width, color='#ff9900')

	ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('ISTA', 'FISTA', 'SCREEN_FISTA', 'CD', 'SCREEN_CD'))

	plt.show()

if __name__ == "__main__":
    main()
