# Enviroment
RL Enviroment simulating a delivery robot, based on Gymnasium Enviroment.<br/>
The field of the environment consists of cells, each of them is instance of Cell object. The Cell object has coordinates, color, target, the robot located on it and neighboring cells (front, right, back, left).
| Action Space |  Box(0, [80, 9], (2,), int64) |
|:---------:|----:|
|    Observation Space    | Discrete(6) | 

The Delivery Robot (Agent) learns navigating to mail station in a grid world, picking mail up and dropping it off at one of nine receiving station.

### Description:
There are nine receiving station in the 9x9 grid world indicated by 1, 2, 3, 4, 5, 6, 7, 8, 9. They are painted yellow. When the episode starts, the robot starts off at a random white cell. The robot drives to the mail station, picks up the mail, drives to the mail's destination (one of the nine specified locations), and then drops off the mail. Mail's destination are given randomly. Once the mail is dropped off, the episode ends (that we can modify after).
### Color Map:
| 9 |   W |    W |    Y |    W |    Y |    W |    Y |    W |    W |   
|:---------:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 |   W |    W |    W |    W |    W |    W |    W |    W |    W |    
| 7 |   Y |    W |    W |    W |    W |    W |    W |    W |    Y |    
| 6 |   W |    W |    W |    W |    W |    W |    W |    W |    W |     
| 5 |   Y |    W |    W |    W |    W |    W |    W |    W |    Y |    
| 4 |   W |    W |    W |    W |    W |    W |    W |    W |    W |   
| 3 |   Y |    W |    W |    W |    W |    W |    W |    W |    Y |   
| 2 |   W |    W |    G |    W |    G |    W |    G |    W |    W |   
| 1 |   W |    W |    W |    W |    W |    W |    W |    W |    W |   
|   |   a |    b |    c |    d |    e |    f |    g |    h |    i |    
### Target Map:
| 9 |    |     |    1 |    |    4 |     |    7 |     |     |   
|:---------:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 |    |     |      |    |      |     |      |     |     |     
| 7 |  3 |     |      |    |      |     |      |     |  6  |     
| 6 |    |     |      |    |      |     |      |     |     |       
| 5 |  2 |     |      |    |      |     |      |     |  5  |         
| 4 |    |     |      |    |      |     |      |     |     |       
| 3 |  1 |     |      |    |      |     |      |     |  9  |        
| 2 |    |     |      |    |      |     |      |     |     |      
| 1 |    |     |      |    |      |     |      |     |     |       
|   |  a  |  b   |  c    | d   |   e   |  f   |   g   |  h   |  i   |       
            
### Actions:
There are 6 discrete deterministic actions:
- 0: move up
- 1: move right
- 2: move down
- 3: move left
- 4: pick up mail
- 5: drop off mail
    
### Observations:
There are 810 discrete states since there are 81 robot positions, 10 possible number of the mail, deliveried by robot (including the case when robot aren't delivering mail). Each state is a 2-dimensional vector, where first coordinate is robot's location (one in 81 cells) and second coordinate is number of the mail, deliveried by robot.

Possible number of the mail, deliveried by robot:
- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

### Info:

``step()`` and ``reset()`` will return an info dictionary that contains 'agent_location', 'mail'
and 'deliveried_mail' containing the cell, where robot locates in, the mail picked up by robot and the mail, dropped off by robot.<br/>
As the steps are deterministic, the probability of the transition is always 1.0

### Rewards
- -1 per step unless other reward is triggered.
- +50 delivering mail.
- -10  executing "pickup" and "drop-off" actions illegally.

# Control System
The control system is optimized by method Q-learning.<br/>
Run:
```
python train_and_evaluate.py

```
to train, evaluate mean reward and save model in 'controller.pkl' file.<br/>
Run:
```
python test.py
```
to watch the control system work. During testing, the control system gives commands to the controlled object according to greed policy. The result of the robot's work process is described in a file "working_process.txt".