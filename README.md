# Cooperation in the PD game in Rewired Scale Free Graphs

## Document introduction:

1. result_program: The programs to run the simulations are in the result program directory
2. simulation_data: \
    a. part1: stores the simulation data for part1 in csv files \
    b. part2: stores the simulation data for part2 in csv files
3. result_figure: The result figures for all the simulation in this project
4. part2
    

The result figures are shown in the result directory.

The programs to run the simulations are in the result program directory, the introduction of
them are also in the file.

The rest of the directories store the data we generated in the process.

## Brief introduction：
For this final project, we are going to study the cooperation in the prisoner's dilemma game in Random scale-free graphs in the means of BA graph.

### Part1: Dynamic feature
For BA network:

Run the simulation, seeing the c, pc, pd, f (explained later in this file) curve against b.
    
For random scale free network:

Rewire BA network to get the random scale free network.
    
Run the simulation, seeing the c, pc, pd, f (explained later in this file) curve against b.

### Part2: Effect of initial condition

Run simulation in both of the network to see the effect of initial condition.


## Brief description of the process

### PART1 Get the relation of c, PC, PD, F against b

1. Create a BA network

2. Get its ADJ matrix

3. Randomly assign the vertices as C (cooperation) or D (defection)

4. For each generation, calculate the gains for each vertex.

   C-C : +1

   C-D : +0

   D-D : +0

   D-C : +b.    (b is from 1.2 to 1.9)

5. Randomly choose a neighbour to decide change strategy or not

   i is the node we look, j is the neighbour

   Pi >= Pj,   keep the original strategy

   Pi <   Pj,   Accepting j's strategy with a probability p
   <img width="1014" alt="Screen Shot 2021-11-12 at 4 57 41 PM" src="https://user-images.githubusercontent.com/58164010/141439624-2967bbe2-2972-4383-8995-95d21577d6bf.png">
​		which means, the p varies for each time we use it, and we need to know the degrees for all the vertices in advance

6. Iteration

   For each time of iteration, the initial gain for each vertex is zero.

   Calculate the  c(t) = c/N for each time,  if dc(t) < 1/sqrt(N), then we can say it reaches steady state.

​		In the thesis, they trained for 5000 times first, they iterated for 1000 times and count the c(t), after reaching steady state, 		iterated 10000 times to see the characterastic in this state.

7. Plot

   n-b curve,

   PC, PD, F curve

### PART 2 The influnce of the initial state

Other things are the same for this part, the only difference is how to assign c or d to the vertices.
<img width="999" alt="Screen Shot 2021-11-12 at 4 57 31 PM" src="https://user-images.githubusercontent.com/58164010/141439656-519ef058-1fcd-4f82-8584-b5e93645b972.png">
We can get a c(t) - b curve and a c means - b curve.


