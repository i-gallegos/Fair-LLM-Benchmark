import sys
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from util import *

def maxSpan(V1,V2):
	maxVal = -2000000
	for i in range(len(V1)):
		for j in range(len(V2)):

			dot = np.abs(np.matmul(V1[i],V2[j].T))
			if dot >= maxVal:
				maxVal = dot
				vec = np.vstack((V1[i]/np.linalg.norm(V1[i]),V2[j]/np.linalg.norm(V2[j])))
	return V1[i]/np.linalg.norm(V1[i]),V2[j]/np.linalg.norm(V2[j])

def proj(u,a):
	return ((np.dot(u,a.T))*u)/(np.dot(u,u))


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


def basis(vec):
	v1 = vec[0]; v2 = vec[1]; 
	v2Prime = v2 - v1*float(np.matmul(v1,v2.T)); 
	v2Prime = v2Prime/np.linalg.norm(v2Prime)
	return v2Prime

# input v1 and v2 are of shape (1,1,d)
def get_basis(v1, v2):
	proj = v1.bmm(v2.transpose(1,2))
	v2_prime = v2 - v1 * proj
	v2_prime = v2_prime/v2_prime.norm()
	return v2_prime


# input v1 and v2 are of shape (1,1,d)
# x of shape (batch_l, seq_l, d)
def rotate(opt, v1, v2, x):
	batch_l, seq_l, d = x.shape
	half_pi = Variable(torch.Tensor([np.pi/2]), requires_grad=False)
	pi = Variable(torch.Tensor([np.pi]), requires_grad=False)
	one_zero = Variable(torch.Tensor([1,0]), requires_grad=False)
	zero_one = Variable(torch.Tensor([0,1]), requires_grad=False)
	if opt.gpuid != -1:
		half_pi = half_pi.cuda(opt.gpuid)
		pi = pi.cuda(opt.gpuid)
		one_zero = one_zero.cuda(opt.gpuid)
		zero_one = zero_one.cuda(opt.gpuid)

	v2_prime = get_basis(v1, v2)
	x_prime = x[:, :, 2:]	# (batch_l,seq_l,d-2)

	proj1 = x.view(-1,1,d).bmm(v1.expand(batch_l*seq_l, 1, d).transpose(1,2))
	proj1 = proj1.view(batch_l, seq_l, 1)
	proj2 = x.view(-1,1,d).bmm(v2_prime.expand(batch_l*seq_l, 1, d).transpose(1,2)) 
	proj2 = proj2.view(batch_l, seq_l, 1)

	x = torch.cat([proj1, proj2], 2)	# (batch_l, seq_l, 2)
	dot = v1.bmm(v2.transpose(1,2))	# (1, 1, 1)
	normalizer = torch.sqrt(1.0 - dot*dot)	# (1,1,1)
	v2 = torch.cat([dot, normalizer], 2)	# (1,1,2)
	v1 = one_zero.view(1,1,2)	# (1,1,2)

	theta = torch.acos(v1.bmm(v2.transpose(1,2)))	# (1,1,1)
	theta_p = half_pi - theta
	norm_x = x/x.norm(p=2, dim=2).unsqueeze(-1)

	prod = v1.expand(batch_l*seq_l, 1, 2).bmm(norm_x.view(-1, 2, 1))
	prod = torch.clamp(prod, -1.0 ,1.0)	# need to clamp it in reality
	phi = torch.acos(prod)
	phi = phi.view(batch_l, seq_l, 1)
	d = zero_one.view(1,1,2).expand(batch_l*seq_l, 1, 2).bmm(norm_x.view(-1, 2, 1))
	d = d.view(batch_l, seq_l, 1)
	
	cond1 = (phi < theta_p) * (d > 0)
	cond1 = cond1.float() * theta * (phi / theta_p)

	cond2 = (phi > theta_p) * (d > 0)
	cond2 = cond2.float() * theta * (pi.view(1,1,1).expand(batch_l, seq_l, 1) - phi)/(pi - theta_p)
	
	cond3 = (phi >= pi-theta_p) * (d < 0)
	cond3 = cond3.float() * theta * (pi.view(1,1,1).expand(batch_l, seq_l, 1) - phi)/theta_p

	cond4 = (phi < pi-theta_p) * (d < 0)
	cond4 = cond4.float() * theta * (phi / (pi - theta_p))

	theta_x = cond1 + cond2 + cond3 + cond4
	theta_x = theta_x.view(-1)


	R = Variable(torch.zeros(batch_l*seq_l, 2,2), requires_grad=False)
	if opt.gpuid != -1:
		R = R.cuda(opt.gpuid)
	R[:, 0,0] = torch.cos(theta_x)
	R[:, 0,1] = -torch.sin(theta_x)
	R[:, 1,0] = -torch.sin(theta_x)
	R[:, 1,1] = torch.cos(theta_x)

	rotated = R.bmm(x.view(-1, 2, 1)).view(batch_l, seq_l, 2)
	return torch.cat([rotated, x_prime], 2)

# U of shape (1, d, d)
# v1 and v2 of shape (1,1,d)
# x of shape (batch_l, seq_l, d)
def correction(opt,U,v1,v2,x):
	batch_l, seq_l, d = x.shape
	U = U.expand(batch_l*seq_l, d, d)
	proj = U.bmm(x.view(batch_l*seq_l, d, 1)).view(batch_l, seq_l, d)
	rotated = rotate(opt,v1, v2, proj)
	corrected = U.transpose(1,2).bmm(rotated.view(-1, d, 1))
	return corrected.view(batch_l, seq_l, d)


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--gpuid', help="Cuda idx", type=int, default = -1)
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
	V2 = np.random.rand(3, opt.dim)
	v1,v2 = maxSpan(V1,V2)
	
	#calculating U once is enough
	U = np.identity(dimensions)
	U = gsConstrained(U,v1,basis(np.vstack((v1,v2))))

	v1 = torch.from_numpy(v1).float().view(1,1,opt.dim)
	v2 = torch.from_numpy(v2).float().view(1,1,opt.dim)
	U = torch.from_numpy(U).float().unsqueeze(0)

	x = torch.randn(50, 300, 300)
	
	#calculating for each word vector x, it's sheared form after contraction
	result = correction(opt,U,v1,v2,x)

	if isnan(result):
		print('*********** nan found')

	print(result.shape)
	print(U.shape)
	print(v1.shape)
	print(v2.shape)
	print(x.shape)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
	x = np.random.rand(300)*4   #put in the vector you want, so, each glove vector we are debiaising
	result = correction(U,v1,v2,x)