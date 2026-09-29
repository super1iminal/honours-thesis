p0 - the distribution of initial states
N - the number of Monte Carlo simulations
y -  a user defined confidence level
T - the time horizon of interest
M - the system model under test

F_0:T = {F_0, F_1, ..., F_T} - the user provided failure regions over the time horizon T of interest
X_0:T <- N Monte Carlo simulated trajectories with model M with initial states drawn from p0
R_0:T = {R(0, L_1), R(1, L_2), ..., R(T, L_T)} - the set of regions defined by X_0:T+1 and F_0:T in equation 11*
    recall that regions change between every time step

t <- 0


equations:

11: 

![alt text](images/image.png)
- how does Rmax relate to Ri? is it just the furthest region?

12: 

![alt text](images/image-1.png)



questions: 
- how many regions in toy car?
- how is the decision of 


notes
start with fixed starting position
at each step, apply the linear operator to figure out the change in speed
covariance matrix for the ellipse

positive definite matrix is ellipse
bisection run to find smallest area ellipse for the right fraction of points

aggreagate points 
vector with number of points in given ellipse out of n
move vector back to host to find bisection
can relaunch with same seeds, also can not care

N(0, sigma) is multidimensional normal

a few tens of ellipses might be a nice number

could do linear apprixmatino when close (after a couple runs)

this is all given eccentricity (relative ratio and orientation) of ellipse

so, perturb these and restart to find another minimum area ellipse


distance can be found with the normal equation

starting interval: ellipse with furthest point and ellipse with closest point

find gradient from derivative

theory:
when close to minimal point, ellipse has similar shape to disturbance Sigma

when you're further away, the fixer is working hard to overpower disturbance, so ellipse shape is more interesting
- 