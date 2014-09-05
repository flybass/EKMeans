EKMeans
=======

An implementation of the Elongated K-Means Algorithm originally authored by Guido Sanguinetti, Jonathan Laidler and Neil D. Lawrence.  Please find their original paper at http://eprints.pascal-network.org/archive/00001544/01/clusterNumber.pdf.

There are a few notable additions.  Following the example of researchers at Carnegie Mellon (please find their work at https://www.cs.cmu.edu/~jgc/publication/PublicationPDF/Automatic_Determination_Of_Number_Of_Clusters_For_Creating_Templates_In_Example-Based_Machine_Translation.pdf), a noise-threshold was added to the algorithm.  

Also, an experimental one-cluster test is added to the algorithm.  As the affinity matrix is the only input, we try to perturb the affinity matrix directly and look at the difference in fit-statistics between 1 cluster and the fit for k>2.  

There is an experimental train routine (to try to find parameters for lambda and epsilon.  This is a greedy search though, and running train rather than some sort of grid-search validation scheme is not recommended.  
