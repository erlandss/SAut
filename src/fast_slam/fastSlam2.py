import numpy as np
import copy as copy
import scipy as sc
from scipy.spatial.transform import Rotation as R
import scipy.linalg as linalg
import random as random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


numParticles = 50

#This may have to be changed to actual arucoIDs
featureIDs =set()
R_robot2camera = R.from_euler("yx",[90,-90],degrees=True).inv()
Q_t =np.array(0.1*np.eye(2))

#Observation model
def h(state,mu):
    theta =state[2]
    position = state[0:2]
    position = np.append(position,0)
    #Make sure z coordinate is in feature mean
    if len(mu)==2:
        mu = np.append(mu,0)
    else:
        mu=np.array(mu)
    R_robot2base = R.from_euler("z",theta,degrees=False)
    R_base2robot = R_robot2base.inv()
    return R_robot2camera.apply((R_base2robot.apply(mu)-position))

def h_inv(measurement,state):
    theta =state[2]
    R_robot2base=R.from_euler("z",theta,degrees=False)
    position = state[0:2]
    position = np.append(position,0)
    return R_robot2base.apply((R_robot2camera.inv().apply(np.array(measurement))+position))[0:2]

#This only returns a selected part (the useful one) of the jacobian
#The full jacobian would also contain information of how the y-measurement, which corresponds to the z-value 
# of the feature in the base coordinate frame
def h_jacobian(state):
    #Find rotation matrix from base frame to camera frame:
    theta = state[2]
    R_robot2base = R.from_euler("z",theta,degrees=False)
    R_base2robot = R_robot2base.inv()
    R_base2camera = R_robot2camera*R_base2robot
    R_b2c_matrix = R_base2camera.as_matrix()
    jacobian = np.array([[R_b2c_matrix[0,0],R_b2c_matrix[0,1]],[R_b2c_matrix[2,0],R_b2c_matrix[2,1]]])
    return jacobian

def drawWeight(weightList):
    total_sum = sum(weightList)
    normalized_weights = [weight / total_sum for weight in weightList]
    draw = random.uniform(0,1)
    previous = 0
    for i in range(len(normalized_weights)):
        if draw >= previous and draw < (previous + normalized_weights[i]):
            return i
        else:
            previous += normalized_weights[i]




class Particle:
    def __init__(self, K: int,x : np.ndarray):
        self.K = K
        self.x = x

        self.features = K*[None]
        for i in range(K):
            self.features[i]={"mean" : np.array([0.0,0.0]), "covariance": np.array(np.eye(2))}

    #Returns feature number i   
    def get_feature(self,i : int):
        return self.features[i]
    

    
class ParticleSet:
    def __init__(self,M:int):
        self.M=M
        self.set =[]
    #appends a particle to the list
    def add(self,particle:Particle):
        self.set.append(particle)

    def remove(self,i:int):
        return self.set.pop(i)
    

initialSet = ParticleSet(numParticles)

# set of randomly distributed in x,y,teta parameters
for i in range(numParticles):
    #Make them random
    initialSet.add(Particle(3,np.array([random.random()*10,random.random()*10,random.random()*2*np.pi])))

beacons = np.array([[3,2],[5,5],[1,4]])
X_t1 = np.array([0,0,0])
samplePositions = np.array([[1,1,0],[2,1,np.pi/4],[3,3,np.pi/4],[3,3,np.pi],[3,3,3*np.pi/2]])
correlations = [0,0,1,2,0]

# print(len(samplePositions))
inputs =[]
observations=[]
for i in range(len(samplePositions)):
    
    newInput = [None]*2
    deltaState = samplePositions[i]-X_t1
    newInput[1]=np.arctan2(deltaState[1],deltaState[0])-X_t1[2]
    newInput[0]=np.sqrt(pow(deltaState[1],2)+pow(deltaState[0],2))
    inputs.append(newInput)

    inputs.append(np.array([0,samplePositions[i,2]-np.arctan2(deltaState[1],deltaState[0])]))
    X_t1 = samplePositions[i]


for i in range(len(samplePositions)):
    observations.append(h(samplePositions[i],beacons[correlations[i]]))

# for i in range(10):
#     print(inputs[i])

def predict(u_t : np.ndarray, p :Particle, delta_t : float):
    (p.x)[2] +=delta_t*u_t[1]
    (p.x)[0] +=delta_t*np.cos((p.x)[2])*u_t[0]
    (p.x)[1] +=delta_t*np.sin((p.x)[2])*u_t[0]




# inputs=[[1,0],[0,1],[1,1]]


#TODO Currently the particles assumes that feature with identifier "k" is at index k
#TODO Fix this^ such that the right feature is acessed without being in index k
#TODO Update step, requires a definition of the observation model h(x,mu)        
def FastSLAM(z_t: np.ndarray, c_t : int ,u_t : np.ndarray, Y_t1 :ParticleSet, delta_t : float):
    yt1 = Y_t1
    weights = np.ones((yt1.M))*(1/yt1.M)
    
    for k in range(yt1.M):
        measurement = z_t
        currentParticle = yt1.set[k]
        #Predict pose
        predict(u_t,currentParticle,delta_t)
        # (currentParticle.x)[2] +=delta_t*u_t[1]
        # (currentParticle.x)[0] +=delta_t*np.cos((currentParticle.x)[2])*u_t[0]
        # (currentParticle.x)[1] +=delta_t*np.sin((currentParticle.x)[2])*u_t[0]
        
        
        


        j=c_t
        
        if j not in featureIDs:
            currentParticle.features[j]["mean"] = h_inv(measurement,currentParticle.x)
            jacobian = h_jacobian(currentParticle.x)
            currentParticle.features[j]["covariance"] = np.matmul(np.matmul(linalg.inv(jacobian),Q_t),np.transpose(linalg.inv(jacobian)))
            featureIDs.add(j)
            #TODO Maybe store particle weights in the set, rather than in this fnc?
        else:
            #predict measurement:
            z_hat = h(currentParticle.x,currentParticle.features[j]["mean"])
            jacobian = h_jacobian(currentParticle.x)
            #Measurement covariance:
            Q = np.matmul(np.matmul(jacobian,currentParticle.features[j]["covariance"]),np.transpose(jacobian))
            Q += Q_t
            #Calculate Kalman gain:
            K = np.matmul(np.matmul(currentParticle.features[j]["covariance"],np.transpose(jacobian)),linalg.inv(Q))
            #Update mean:
            z_hat =np.array([z_hat[0],z_hat[2]])
            
            measurement =np.array([measurement[0],measurement[2]])
            currentParticle.features[j]["mean"] += np.matmul(K,(measurement-z_hat))
            #Update covariance:
            currentParticle.features[j]["covariance"] = np.matmul((np.eye(2)-np.matmul(K,jacobian)),currentParticle.features[j]["covariance"])
            #update weight/importance factor:
            weights[k] = np.power(linalg.det(2*np.pi*Q),-0.5)*np.exp(-0.5*np.matmul(np.transpose((measurement-z_hat)),np.matmul(linalg.inv(Q),(measurement-z_hat))))
            


        # for i in featureIDs:
        #     #leave unchanged
        #     print()
    
    Yt = ParticleSet(yt1.M)
    for i in range(yt1.M):
        k = drawWeight(weights)
        p =copy.deepcopy(yt1.set[k])
        Yt.add(p)
    #print("\n")

    return Yt




# Set general plot parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Create figure and add axes
fig = plt.figure(figsize=(6, 6))  # Set the figure size to be square
ax = fig.add_subplot(111)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

# Create variable reference for particles
particles = ax.scatter([], [], c='blue', s=1)


# starts with the initial set
particlePoints = initialSet
# stores all sets - one set for each input of linear velocity or angular velocity (in this case, 10 inputs)
allSets = [particlePoints]
# for i in range(numParticles):
#     print(allSets[0].set[i].x)

# goes through every input
for i in range(5):

    # previous (t-1) particle set
    particlePoints = allSets[-1]

    # predicts particle set for even inputs - no measurement
    # adds particles to the set
    for j in range(numParticles):
        predict(inputs[2*i],particlePoints.set[j],1)
        print(particlePoints.set[j].x)
        print("\n")

    # append new predicted particle sets to set of sets
    allSets += [particlePoints]

    # previous (t-1) particle set
    particlePoints = allSets[-1]

    # predict and update particle sets for odd inputs - measurement
    newSet = FastSLAM(observations[i],correlations[i],inputs[2*i+1],particlePoints,1)

    allSets.append(newSet)
    particlePoints = newSet




# Function to update the plot for each frame
def update_plot(frame):
    ax.clear()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    # we only want the list of x,y to plot
    currentSet = allSets[frame].set

    # Extract x, y values from particle set
    x_values = [particle.x[0] for particle in currentSet]
    y_values = [particle.x[1] for particle in currentSet]

    # Update the particle set
    particles = ax.scatter(x_values, y_values, c='blue', s=1)

    # Set square aspect ratio and frame around the plot
    ax.set_aspect('equal')  # Set aspect ratio to be equal
    ax.spines['left'].set_visible(False)  # Hide left spine
    ax.spines['bottom'].set_visible(False)  # Hide bottom spine
    ax.spines['right'].set_visible(False)  # Hide right spine
    ax.spines['top'].set_visible(False)  # Hide top spine

    # Add a frame around the plot
    ax.axhline(0, color='black', linewidth=2)  # Horizontal line at y=0
    ax.axhline(10, color='black', linewidth=2)  # Horizontal line at y=1
    ax.axvline(0, color='black', linewidth=2)  # Vertical line at x=0
    ax.axvline(10, color='black', linewidth=2)  # Vertical line at x=1

numFrames = 10 

# Update the plot for each frame
for frame in range(numFrames):
    update_plot(frame)
    plt.pause(0.5)  # Pause between frames

# Display the plot
plt.show()
