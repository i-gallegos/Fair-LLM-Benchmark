import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp 
import sys
import argparse


def readFile(fileName):
	f1 = open(fileName,'r')
	f1.readline()
	data = f1.readlines()
	return data


def process(data):
	A = np.zeros((len(data),3))
	for i in range(len(data)):
		a = data[i].strip().split(',')
		A[i][0] = float(a[4]);
		A[i][1] = float(a[5]);
		A[i][2] = float(a[6]);

	return A


def netNeutral(A):
	return sum(A)/len(A)

def sortedDist(A):
	tempA = np.asarray(A)
	sortedArg = np.argsort(tempA)
	tempA = np.sort(tempA)
	return sortedArg, tempA

def sortedArgs(l1,val,l2,lE,lC):
	L = [0]*len(l1)
	for i in range(len(l1)):
		if lE[l1[i]] > lC[l1[i]]:
			L[l1[i]] = (l2[i],'e',lE[i],val[i])
		else:
			L[l1[i]] = (l2[i],'c',lC[i],val[i])
	return L


def sortedFunc(list1,tupleList,listE,listC):
	ind = np.argsort(list1); 
	return 'Done'

def KLD(A,i,j):
	return sp.stats.entropy(A[i],A[j])

def counter(A,threshold):
	pass

def cdf(A,arr):
	for i in range(1,len(arr)):
		arr[i] += arr[i-1]
	return np.asarray(arr)/len(A)

def plotting(val):
	plt.plot(range(len(val)), val)
	plt.xlabel('Sorted Tuples', fontsize = '20')
	plt.ylabel('P_neutral', fontsize = '18')
	plt.show()

def structure(data):
	for i in range(len(data)):
		data[i] = data[i].strip().split(',')
	return data

def filtering(data):
	newData = []
	for i in range(len(data)):
		newData.append(data[i])
	return newData


def club(data):
	newData = dict(); newList1=[]; newListN=[]; newListE =[]; newListC= []
	for i in range(len(data)):
		if (data[i][0],data[i][1]) not in newData:
			newData[(data[i][0],data[i][1])] = [float(data[i][4]),float(data[i][5]),float(data[i][6]),1.0]
		else :
			newData[(data[i][0],data[i][1])][0] += float(data[i][4])
			newData[(data[i][0],data[i][1])][1] += float(data[i][5])
			newData[(data[i][0],data[i][1])][2] += float(data[i][6])
			newData[(data[i][0],data[i][1])][3] += 1.0
	for key in newData:
		newList1.append(key); newListN.append(newData[key][1]/newData[key][3]); newListE.append(newData[key][0]/newData[key][3]); newListC.append(newData[key][2]/newData[key][3]); 
	return newList1, newListN, newListE, newListC


def fracNeutral(l, alpha):
	counter = 0.0
	for i in range(len(l)):
		if l[i] > alpha:
			counter = counter + 1.0
	return counter/len(l)


def ksTest(d1,d2):
	return sp.stats.ks_2samp(d1,d2)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', help="Path to the baseline prediction file", default="")

def main(args):
	opt = parser.parse_args(args)
	
	data = readFile(opt.data)

	A = process(data)
	#plt.plot()
	
	data = structure(data)
	dataL1, dataL2 ,dataL3 , dataL4 = club(data)
	print('kstest',ksTest(cdf(A, np.sort(dataL2))))

	args = sortedArgs(sortedDist(dataL2)[0],dataL2,dataL1,dataL3,dataL4)

	print('Net Neutral : ', netNeutral(dataL2),'Threshold = 0.5 : ', fracNeutral(dataL2,0.5),'Threshold = 0.7 : ', fracNeutral(dataL2,0.7))


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))