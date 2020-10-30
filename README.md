# OD-MatrixEstimation
this is my undergrad thesis project regarding estimating Origin Destination Matrices using convex optimization and sparse reconstruction techniques. There is a brief presentation available [here](Slides%202.pdf)
File 'Test.ipynb' contains a synthesis simulation.  

## Introduction
There is an annual questionairre where every household should fill out which entails questions regarding the place that you live in and places that you frequently visit. The reason is that Urban planning requires data regarding demand between distinct regions of the city. For instance, around 6-7AM people will commute from their houses to their work places and it is crucial to have this information in the regional level so that we'd be able to control the traffic optimally. The goal of this project is to elevate this mannual process into a more robust data oriented process. The data that we intend to use is a traffic index in the level of streets scraped from google maps. 

*Challenges*:
-- Traffic index provided by google mpas is discrete so we lose a bit of information through discretization which makes our estimation harder.
-- Traffic is an indirect aggregated consequnce of this origin destination demand. Thus, we are dealing with an inverse problem where we need to impose conditions in order to be able to achieve this. For instance, usual compressed sensing assumptions might be required so that the problem would not be ill-posed. 

The algorithm was tested on a synthesis dataset where the assumptions were met. 

![results](results.png)

## Method
First we generated a rank one origin destination demand matrix and then affect the matrix with a distance matrix. We 
used the gravity model by which I meant that every region has two masses, one for retraction and one for attraction and we 
have assumed that the demand satisfy the newtonian gravity law framework. Subsequently, we have solved the 
user [equilibrium](https://en.wikipedia.org/wiki/John_Glen_Wardrop) via a projection free method. You can find main method 
written in the notebook with more clarity! 

