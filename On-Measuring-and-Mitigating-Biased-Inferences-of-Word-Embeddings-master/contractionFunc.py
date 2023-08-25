import sys
import argparse
import numpy as np

def maxSpan(V1,V2):
	maxVal = -2000000
	for i in range(len(V1)):
		for j in range(len(V2)):

			dot = np.matmul(V1[i],V2[j].T)
			if dot >= maxVal:
				maxVal = dot
				vec = np.vstack((V1[i]/np.linalg.norm(V1[i]),V2[j]/np.linalg.norm(V2[j])))
	return V1[i]/np.linalg.norm(V1[i]),V2[j]/np.linalg.norm(V2[j])

def proj(u,a):
	return ((np.dot(u,a.T))*u)/(np.dot(u,u))

def basis(vec):
	v1 = vec[0]; v2 = vec[1]; 
	v2Prime = v2 - v1*float(np.matmul(v1,v2.T)); 
	v2Prime = v2Prime/np.linalg.norm(v2Prime)
	return v2Prime

def gsConstrained(matrix,v1,v2):
	v1 = np.asarray(v1).reshape(-1)
	v2 = np.asarray(v2).reshape(-1)
	u = np.zeros((np.shape(matrix)[0],np.shape(matrix)[1]))
	u[0] = v1
	u[0] = u[0]/np.linalg.norm(u[0])
	u[1] = v2 - proj(u[0],v2)
	u[1] = u[1]/np.linalg.norm(u[1])
	for i in range(0,len(matrix)-2):
		p = 0.0
		for j in range(0,i+2):	
			p = p + proj(u[j],matrix[i])
		u[i+2] = matrix[i] - p
		u[i+2] = u[i+2]/np.linalg.norm(u[i+2])
	return u


def rotation(v1,v2,x):
	v1 = np.asarray(v1).reshape(-1)
	v2 = np.asarray(v2).reshape(-1)
	x = np.asarray(x).reshape(-1)

	v2P = basis(np.vstack((v1,v2)))
	xP = x[2:len(x)]

	x = (np.dot(x,v1),np.dot(x,v2P)) 
	v2 = (np.matmul(v2,v1.T),np.sqrt( 1 - (np.matmul(v2,v1.T)**2)))
	v1 = (1,0)
	thetaX = 0.0
	theta = np.arccos(np.dot(v1,v2))
	thetaP = (np.pi/2.0) - theta
	phi = np.arccos(np.dot(v1,x/np.linalg.norm(x)))
	d = np.dot([0,1],x/np.linalg.norm(x))
	if phi<thetaP and d>0:
		thetaX = theta*(phi/thetaP)
	elif phi>thetaP and d>0:
		thetaX = theta*((np.pi - phi)/(np.pi - thetaP))
	elif phi>=np.pi - thetaP and d<0:
		thetaX = theta*((np.pi-phi)/thetaP)
	elif phi<np.pi - thetaP and d<0:
		thetaX = theta*(phi/(np.pi-thetaP))
	R = np.zeros((2,2))
	R[0][0] = np.cos(thetaX); R[0][1] = -np.sin(thetaX)
	R[1][0] = -np.sin(thetaX); R[1][1] = np.cos(thetaX)
	return np.hstack((np.matmul(R,x),xP))

def correction(U,v1,v2,x):
	return np.matmul(U.T,rotation(v1,v2,np.matmul(U,x)))

def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--v1', help="Path to dir 1 vector", default = "")
	parser.add_argument('--v2', help="Path to dir 2 vector", default = "")
	parser.add_argument('--dim', help="dimension of vector", type=int, default = 300)
	opt = parser.parse_args(arguments)

	#Running it
	dimensions = opt.dim

	#loading the two direction vector files
	#V1 = np.asmatrix(np.loadtxt(opt.v1))
	#V2 = np.asmatrix(np.loadtxt(opt.v2))
	V1 = np.random.rand(1, opt.dim)
	V2 = np.random.rand(1, opt.dim)
	v1,v2 = maxSpan(V1,V2)
	
	#calculating U once is enough
	U = np.identity(dimensions)
	U = gsConstrained(U,v1,basis(np.vstack((v1,v2))))

	x = np.random.rand(300)   #put in the vector you want, so, each glove vector we are debiaising
	
	#calculating for each word vector x, it's sheared form after contraction
	result = correction(U,v1,v2,x)

	print(result.shape)
	print('U.shape', U.shape)
	print(v1.shape)
	print(v2.shape)
	print(x.shape)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
	x = np.random.rand(300)   #put in the vector you want, so, each glove vector we are debiaising
	result = correction(U,v1,v2,x)
