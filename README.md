# SPX-graph-layout
Stress-Plus-X (SPX) graph layout

Stress, edge crossings, and crossing angles play an important role in the quality and readability of graph drawings. Most standard graph drawing algorithms optimize one of these criteria which may lead to layouts that are deficient in other criteria. We introduce an optimization framework, Stress-Plus-X (SPX), that simultaneously optimizes stress together with several other criteria: edge crossings, minimum crossing angle, and upwardness (for directed acyclic graphs). SPX achieves results that are close to the state-of-the-art algorithms that optimize these metrics individually. SPX is flexible and extensible and can optimize a subset or all of these criteria simultaneously. Our experimental analysis shows that our joint optimization approach is successful in drawing graphs with good performance across readability criteria.

###Instructions

`python3 spx.py`

Requires the following packages
  * networkx
  * numpy
  * matplotlib
  * scipy
  * CPLEX with python API [https://www.ibm.com/analytics/cplex-optimizer]
