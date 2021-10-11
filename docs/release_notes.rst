.. role:: small
.. role:: smaller
.. role:: noteversion

Version 0.4.4 :small:`11 October 2021`
---------------------------------------
Fixed a bug regarding the distance metric usage in cost calculation.

Version 0.4.3 :small:`20 April 2021`
---------------------------------------
Fixed bugs. Added self consistency analysis and updated tutorials.

Version 0.4.2 :small:`03 April 2021`
---------------------------------------
Improved package structure, fixed minor performace issues, and fixed bugs. Added two new tutorials (corti & osteosarcoma) and 
updated the previous tutorials with validation analyses.

Version 0.4.1 :small:`24 August 2020`
---------------------------------------
Changed the package structure and run flow of the scripts. Added anndata and scanpy support. Updated tutorials and implemented
basic target geometries.

Version 0.3.11 :small:`27 April 2020`
---------------------------------------
Moran's I algorithm for spatially informative genes is implemented and removed pysal dependency.

Version 0.3.10 :small:`07 February 2020`
---------------------------------------
Added Moran's I algorithm to detect spatially informative genes.

Version 0.3.7 :small:`29 October 2019`
---------------------------------------
Updated computation of shortest paths that singificantly reduces
running time.

Version 0.3.5 :small:`13 June 2019`
---------------------------------------
Fixed a bug that was prone to produce infinities during reconstruction.
Improved plotting functions and added new ones for plotting mapped cells.

Version 0.3.4 :small:`27 February 2019`
---------------------------------------
novoSpaRc reconstructs single-cell gene expression without relying on existing
reference markers and makes great use of such information if available.
