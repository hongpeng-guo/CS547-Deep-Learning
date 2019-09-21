import numpy as np
import scipy 


def sigmoid(z):

	return 1./(1. + np.exp(-z))


def im2col(img, ky, kx, stride=1):
	N, H, W = img.shape
	out_h = (H - ky) // stride + 1
	out_w = (W - kx) // stride + 1
	col = np.empty((N * out_h * out_w, ky * kx))
	outsize = out_w * out_h
	for y in range(out_h):
		y_min = y * stride
		y_max = y_min + ky
		y_start = y * out_w
		for x in range(out_w):
			x_min = x * stride
			x_max = x_min + kx
			col[y_start+x::outsize, :] = img[:, y_min:y_max, x_min:x_max].reshape(N, -1)
	return col

'''
def conv(X, K, stride=1, padding=0):
	FN, H_k, W_k, C = K.shape
	N, H, W = X.shape
	col = im2col(X, H_k, W_k, stride)
	z = np.dot(col, K.reshape(K.shape[0], -1).transpose())
	z = z.reshape(N, z.shape[0] / N, -1)
	out_h = (H - ksize) // stride + 1
	return z.reshape(N, out_h, -1 , FN)
'''

def conv(X, K, stride=1, padding=0):
	H_k, W_k, C = K.shape
	N, H, W = X.shape
	col = im2col(X, H_k, W_k, stride)
	z = np.dot(col, K.reshape(H_k * W_k, -1))
	out_h = (H - H_k) // stride + 1
	return z.reshape(N, out_h, -1 , C)[0,:,:,:]



def feed_forward(X, model):

	'''
	Z[:,:,p] = conv(X, K[:,:,p])
	H[:,:,p] = sigmoid(Z[:,:,p])
	U[k] = W[K,:,:,:] @ H[:,:,:] + b[k]
	R = softmax(U)
	'''
	cache = {}

	cache["Z"] = conv(X, model["K"])
	cache["H"] = sigmoid(cache["Z"])
	cache["U"] = np.matmul(model["W"].reshape(model["W"].shape[0], -1), np.expand_dims(cache["H"].reshape(-1), axis=1)) + model["b"]
	cache["R"] = np.exp(cache["U"]) / np.sum(np.exp(cache["U"]), axis=0)

	return cache



def back_propagate(X, Y, model, cache):
	
	'''
	db = rou_to_U := cache["R"] - Y
	delta := np.matmul(db.T, W.reshape[i, -1]).reshape(i, j, p)
	dK = conv(X, delta * sigma_prime(Z))
	dW = np.matmul(db, np.expand_dims(cache["H"], axis=0))
	'''
	batch_size = X.shape[1]
	
	db = cache["R"] - Y

	k, i, j, p = model["W"].shape
	delta = np.matmul(db.T, model["W"].reshape(k, -1)).reshape(i, j, p)

	dK = conv(X, delta * sigmoid(cache["Z"]) * (1 - sigmoid(cache["Z"])))
	dW = np.matmul(db, cache["H"].reshape(1,-1)).reshape(k, i, j, p)

	grads = {"b": db, "W": dW, "K": dK}

	return grads

