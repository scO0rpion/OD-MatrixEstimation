# OD-MatrixEstimation
This is a repository containing my undergrad thesis project regarding origin destination matrices estimation using convex optimization and sparse reconstruction techniques. There is a brief presentation available [here](Slides%202.pdf)
File 'Test.ipynb' contains a synthetis simulation.  

## Introduction
There is an annual questionairre where every household should fill out which entails questions regarding the places that you frequently visit and currently reside in. This information is then used for urban planning by estimating the demand between distinct regions of the city. For instance, around 6-7AM people will commute from their houses to their work places (demand) and the goal is to build an expectation of the traffic as a consequence of this demand.  The goal of this project is to elevate this mannual process into a more robust data oriented process. The data that we intend to use is a traffic index scraped from google maps on a street level. 

**Challenges**:
* Traffic index provided by google mpas is discrete so we lose a bit of information through discretization which makes our estimation problem more difficult.
* Traffic is an indirect aggregated consequnce of this origin-destination demand. Moreover, we are dealing with an inverse problem for which identifiability without assumptions on the data structure may be hopeless. Thus, we need to impose some conditions in order to be able to resolve this inverse problem. Usual compressed sensing assumptions might be required so that the problem would not be ill-posed. 

The algorithm introduced below was tested on a synthetic dataset. 

![results](results.png)

## Method
We generated a rank one origin destination demand matrix and then affect the matrix with a distance matrix. We 
use the gravity model that us every region has two masses, one for retraction and one for attraction, and 
assumed the demand satisfy the newtonian gravity law framework-- two regions are attracted to each other based on multiplication of their associated masses and inversely proportional to some measure of distance betweem them. Subsequently, we have solved the 
[use equilibrium](https://en.wikipedia.org/wiki/John_Glen_Wardrop) via a projection free method.

