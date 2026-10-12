# SPDX-License-Identifier: MulanPSL-2.0
"""Independent NumPy mathematical references; inputs already target-quantized."""
import numpy as np


def axpby(x,y,alpha=2.0,beta=-0.5):return alpha*x+beta*y

def identity(x):return np.array(x,copy=True)

def sum_max(x):return np.asarray(np.sum(x)),np.asarray(np.max(x) if x.size else -np.inf)

def pair(x):return x+1,x*2

def fill(n=17,value=2.0,dtype='float32'):return np.full((n,),value,dtype=dtype)

def matmul(a,b):return np.matmul(a,b)

def softmax(x,dim=-1):
    if x.size==0:return np.empty_like(x)
    e=np.exp(x-np.max(x,axis=dim,keepdims=True));return e/np.sum(e,axis=dim,keepdims=True)

def inplace_add(x,value=1.0):return x+value

def optional_bias(x,bias=None):return x.copy() if bias is None else x+bias

def tensor_list_sum(xs):
    if not xs:raise ValueError('tensor list cannot be empty')
    return np.sum(np.stack(xs),axis=0)

def nonzero(x):return np.transpose(np.nonzero(x)).astype(np.int64)

def fail(x):raise ValueError('controlled expected execution failure')

def wrong(x):return x+1

def cpu_pair(x):return (x+np.asarray(1,dtype=x.dtype),np.arange(x.size,dtype=np.int64).reshape(x.shape))

def cpu_pair_reference(x):return (x+1,np.arange(x.size,dtype=np.int64).reshape(x.shape))
