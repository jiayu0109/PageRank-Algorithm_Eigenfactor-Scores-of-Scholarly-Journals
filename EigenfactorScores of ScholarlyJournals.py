#!/usr/bin/env python
# coding: utf-8

# In[]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import scipy.linalg as la


# In[]:
### 1.1 - 1.2 Creating an Adjacency Matrix
# In the first section, I'll try to replicate example raw adjacency matrix (Z) on the problem set file. 
# First I create the function to turn raw data into matrix, because I'll have to use it on the big dataset afterwards

# Add a vertex to the set of vertices and the graph
def add_vertex(v):
  global graph
  global vertices_no
  global vertices
  if v in vertices:
    print("Vertex ", v, " already exists")
  else:
    vertices_no = vertices_no + 1
    vertices.append(v)
    if vertices_no > 1:
        for vertex in graph:
            vertex.append(0)
    temp = []
    for i in range(vertices_no):
        temp.append(0)
    graph.append(temp)

# Add an edge between vertex v1(cited journal) and v2(citing journal) with edge weight e(citing times)
def add_edge(v1, v2, e):
    global graph
    global vertices_no
    global vertices
    if v1 not in vertices:
        print("Vertex ", v1, " does not exist.")
    elif v2 not in vertices:
        print("Vertex ", v2, " does not exist.")
# Since this code is not restricted to a directed or  an undirected graph, an edge between v1 v2 does not imply that an edge exists between v2 and v1
    else:
        index1 = vertices.index(v1)
        index2 = vertices.index(v2)
        graph[index1][index2] = e

# Start to import the example data manually
vertices = []
vertices_no = 0
graph = []
# Add vertices to the graph(there're 6 journals in the example dataset)
add_vertex(1)
add_vertex(2)
add_vertex(3)
add_vertex(4)
add_vertex(5)
add_vertex(6)
# Add the edges between the vertices by specifying cited journal, citing journal and citing times.
add_edge(1, 1, 1)
add_edge(1, 3, 2)
add_edge(1, 5, 4)
add_edge(1, 6, 3)
add_edge(2, 1, 3)
add_edge(2, 3, 1)
add_edge(2, 4, 1)
add_edge(3, 1, 2)
add_edge(3, 3, 4)
add_edge(3, 5, 1)
add_edge(4, 3, 1)
add_edge(4, 6, 1)
add_edge(5, 1, 8)
add_edge(5, 3, 3)
add_edge(5, 5, 5)
add_edge(5, 6, 2)
print("Internal representation: ", graph)

# turn the graph into the array
m = np.array(graph)
print (m)


# In[]:
### 1.3 Modifying the Adjacency Matrix
# Set the diagonal to zero
np.fill_diagonal(m, 0)
print(m)

# Normalize the columns. This matrix is H.
from sklearn import preprocessing
H = preprocessing.normalize(m, axis=0, norm='l1')
print(H)

# In[]:
### 1.4 Identifying the Dangling Nodes >> "d" here
test = m.sum(axis=0)
d = [1 if i == 0 else 0 for i in test]
d


# In[]:
### 1.5 Calculating the Influence Vector
# 1.5-1 Article Vector
A_Vector = np.array([[3, 2, 5,1, 2, 1]]).T
A_Vector = preprocessing.normalize(A_Vector, axis=0, norm='l1')
A_Vector

# 1.5-2 Initial start vector π(0)
num_column = 6
i_vector = np.arange(num_column)
i_vector = [1/num_column for i in i_vector]
i_vector = np.array([i_vector]).T
i_vector

# 1.5-3 Calculating the influence vector π∗
# Apply the "P=αH′ +(1−α)a.eT"
α = 0.85
ε = 0.00001
current = i_vector
def checkpoint(current):
    n = 0
    diff = 1
    while diff > ε:
        p = α*np.matmul(H,current)+((α*np.matmul(d,current))+(1-α))*A_Vector
        diff = np.sum(np.absolute(p - current))
        n = (n+1)
        current = p
    print(n)
    return(p)
inf_v = checkpoint(current)
# It takes 18 loops to get the converged influence vector

# In[]:
### 1.6 Calculating Eigenfactor (EFi)
# To get the EF with example dataset, the final step is to dot product the H matrix and the influence vector π∗
# and normalized to sum to 1, and then multiplied by 100 
EF = 100*(preprocessing.normalize(np.dot(H,inf_v), axis=0, norm='l1'))
print(EF)
# The result is the same as the answer in the provided pdf file.


# In[]
### Apply to the real data set with 10,747 unique journals (nodes)
# read the REAL file
real_lists = []
with open("links.txt","r") as f:
    for line in f:
        inner_list = [int(elt.strip()) for elt in line.split(',')]
        real_lists.append(inner_list)
        
# test whether I read the file successfully
print(real_lists[:3])

# Create an Adjacency Matrix with this data
vertices = []
vertices_no = 0
graph = []
num_node = 10747
def run_vertex(node):
    for i in range(node):
        add_vertex(i+1)

def run_edge(tt):
    for i in range(len(tt)):
        add_edge(tt[i][1],tt[i][0],tt[i][2]) # (cited journal, citing journal, num of citation)
run_vertex(num_node)
run_edge(real_lists)
# Note: it takes some time to build the graph.

# In[]
m = np.array(graph)

## Modify and normalize the Adjacency Matrix
np.fill_diagonal(m, 0)
H = preprocessing.normalize(m, axis=0, norm='l1')

## Dangling Nodes
test = m.sum(axis=0)
d = [1 if i == 0 else 0 for i in test]

## Article Vector
# As specified in the assignment, I assume all articles publish one paper (i.e., a uniform teleport).
# and normalize it.(so basically 1/10747 for every journal)
num_column = 10747
A_Vector = np.arange(num_column)
A_Vector = [1/num_column for i in A_Vector]
A_Vector = np.array([A_Vector]).T

## Initial start vector π(0)
num_column = 10747
i_vector = np.arange(num_column)
i_vector = [1/num_column for i in i_vector]
i_vector = np.array([i_vector]).T

## Calculate the influence vector π∗
# Apply the "P=αH′ +(1−α)a.eT"
α = 0.85
ε = 0.00001
current = i_vector
def checkpoint(current):
    n = 0
    diff = 1
    while diff > ε:
        p = α*np.matmul(H,current)+((α*np.matmul(d,current))+(1-α))*A_Vector
        diff = np.sum(np.absolute(p - current))
        n = (n+1)
        current = p
    print(n)
    return(p)
inf_v = checkpoint(current)
# It takes 32 loops to get the converged influence vector

## Calculating Eigenfactor (EFi)
EF = 100*(preprocessing.normalize(np.dot(H,inf_v), axis=0, norm='l1'))
ANS = []

# Here I want to find which are top 20 journals by locating them. And I add 1 because 
# python's index starts with 0.
for n in EF:
    ANS.append(float(n));
Answer = np.array(ANS)
n = 20
indices = ((-Answer).argsort()[:n])+1
print(indices)

# Here I want to find top 20 journals' score from high to low
rslt = Answer[np.argsort(Answer)][-20:]
print(np.flip(rslt))


# calculate the time
import time
start = time.time()
m = np.array(graph)
np.fill_diagonal(m, 0)
H = preprocessing.normalize(m, axis=0, norm='l1')
test = m.sum(axis=0)
d = [1 if i == 0 else 0 for i in test]

num_column = 10747
A_Vector = np.arange(num_column)
A_Vector = [1/num_column for i in A_Vector]
A_Vector = np.array([A_Vector]).T
i_vector = np.arange(num_column)
i_vector = [1/num_column for i in i_vector]
i_vector = np.array([i_vector]).T

inf_v = checkpoint(current)
EF = 100*(preprocessing.normalize(np.dot(H,inf_v), axis=0, norm='l1'))
ANS = []
for n in EF:
    ANS.append(float(n));
Answer = np.array(ANS)
n = 20
indices = ((-Answer).argsort()[:n])+1
rslt = Answer[np.argsort(Answer)][-20:]

end = time.time()
print(f"Runtime of the program is {end - start}")


# In[]
### Final Answer
# (a) report the scores for the top 20 journals 
# ANS: [1.44816186 1.41277676 1.23505516 0.67956036 0.66493231 
#       0.63466379 0.57725665 0.48085309 0.47779725 0.43976428 
#       0.42975632 0.38623312 0.38515252 0.37961394 0.37281859 
#       0.33033304 0.32752883 0.31928752 0.31679909 0.3112832 ]
# Top 20 journals are 
#      [4408 4801 6610 2056 6919 
#       6667 4024 6523 8930 6857 
#       5966 1995 1935 3480 4598 
#       2880 3314 6569 5035 1212]

# (b) report the time it took to run your code on this real network 
# ANS: 24.57 seconds(from adjusting the metrix to get the final result)

# (c) report the number of iterations it took to get to your answer
# ANS: 32 times

