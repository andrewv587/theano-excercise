#!/bin/python
#Filename:theano1.py
#Function: add two scalar in theano
#Author:Huang Weihang
#Email:huangweihang14@mails.ucas.ac.cn
#Data:2016-09-01
import numpy as np
import theano.tensor as T
from theano import In
from theano import function
from theano import shared
import numpy as np
from theano import pp
from theano import scan

X=T.dmatrix('X')
Y=T.dmatrix('Y')
results,updates = scan(lambda y,x:x+y,sequences=[Y],outputs_info=[X])
f=function([X,Y],results)
x=np.random.rand(3,4)
y=np.ones((3,4))
print x
print y
print f(x,y)

#v=T.dmatrix('v')
#A=T.dmatrix('a')
#av=A.dot(v)
#va=v.dot(A)
#f=function([A,v],[av,va])
#print f([[1,2],[3,4]],[[1,2],[1,2]])


#x=T.dscalar('x')
#z=T.dscalar('z')
#y=x**2+z
#gx,gz=T.grad(y,[x,z])
#print pp(gz)
#f=function([x,z],[gx,gz])
#print f(4,1)


#rng=np.random
#N=400
#feats=784
#
##generate label
#D=(rng.rand(N,feats),rng.randint(0,2,N))
##print D
#
##generate model
#x=T.dmatrix('x')
#y=T.dvector('y')
#
#w=shared(rng.randn(feats),name='w')
#b=shared(0.0,name='b')
#
#print "Initial model"
#print w.get_value()
#print b.get_value()
#
#hx=1/(1+T.exp(-T.dot(x,w)-b))
#prediction = hx>0.5
#xcent = -y*T.log(hx)-(1-y)*T.log(1-hx)
#cost = xcent.mean()+0.01*(w**2).sum()
#gw,gb=T.grad(cost,[w,b])
#
#train=function([x,y],[prediction,xcent],updates=[(w,w-0.1*gw),(b,b-0.1*gb)])
#for i in range(10000):
#    print i
#    pred,err = train(D[0],D[1])
#
#print "final model"
#print w.get_value()
#print b.get_value()
#
#print "target values for D:"
#print D[1]
#print "prediction on D:"
#print pred
    

#x=T.dscalar("x")
#y=T.dscalar("y")
#z=x+y
#f=function([x,y],z)
#print f(2,3)

#a=T.dvector('x')
#b=T.dvector('y')
#out = a**2+b**2+2*a*b
#f=function([a,b],out)
#print f([1,2],[4,5])

#x=T.dmatrix('x')
##s=1/(1+T.exp(-x))
#s=(1+T.tanh(x/2))/2
#f=function([x],s)
#print f([[0,1],[-1,-2]])

#a,b=T.dmatrices('a','b')
#diff=a-b
#abs_diff=abs(diff)
#diff_squared = (diff)**2
#f=function([a,In(b,value=[[0,1],[2,3]],name='nb')],[diff,abs_diff,diff_squared])
#print f([[1,1],[1,1]],nb=[[0,1],[2,3]])
#print f([[1,1],[1,1]])

#state=shared(0)
#inc = T.iscalar("inc")
#accumlator=function([inc],state,updates=[(state,state+inc)])
#print accumlator(1)
#print state.get_value()
#state.set_value(10)
#print accumlator(2)
#print state.get_value()

#state=shared(0)
#inc=T.lscalar('inc')
#inc1=T.lscalar('inc')
#skip_shared=state+1+inc
##f=function([inc],skip_shared)
#f=function([inc],skip_shared,givens=[(state,inc)])
#print f(1)

#from theano.tensor.shared_randomstreams import RandomStreams
#srng = RandomStreams(seed=234)
#rv_u = srng.uniform((2,2))
#rv_n = srng.normal((2,2))
#f = function([], rv_u)
#g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
#nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
#print f()
#print f()
#print g()
#print g()
