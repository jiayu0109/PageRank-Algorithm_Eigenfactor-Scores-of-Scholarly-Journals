# Eigenfactor-Scores-of-Scholarly-Journals
Calculating Eigenfactor Scores based on PageRank Algorithm


## 1. Data Input
Originally we need two files: 

(1) Journal Citation File: how often each journal cites all other journals

(2) Article File: the number of articles that each journal produces in past five years

Yet it's a practice for me to replicate the example further down in this document, so I'll manually create a matrix instead.

The constants below will be used in replicating the algorithm.

• Alpha constant (α = 0.85)

• Epsilon constant (ε = 0.00001)

## 2. Creating an Adjacency Matrix
The journal citation network can be represented as an adjacency matrix.

The Zij-th entry indicates the number of times that articles published in journal j during the period cite articles in journal i published during the target window.

## 3. Modifying the Adjacency Matrix
(1) Set the diagonal of Z to zero(so that each journal won't receive the credit of self-citation)

(2) Normalize the columns of the matrix Z

## 4. Identifying the Dangling Nodes
When journals don't cite any other journals, they are called dangling nodes.

## 5. Calculating the Stationary Vector
- Calculate Article Vector
- Calculate Initial start vector
- Calculate the influence vector π∗.

## 6. Calculationg the EigenFactor (EF) Score
How to: 
- Dot product of the H matrix and the influence vector π∗ 
- Normalized to sum to 1
- Multiplied by 100 (convert the values to percentages)

