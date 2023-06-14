import numpy as np
import copy as copy
import scipy as sc
from scipy.spatial.transform import Rotation as R
import scipy.linalg as linalg
import random as random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#This may have to be changed to actual arucoIDs
featureIDs =set()
R_robot2camera = R.from_euler("yx",[90,-90],degrees=True).inv()
Q_t =np.array(0.001*np.eye(2))
variances = np.array([0.01,0.001,0])
missedObservations = set()
hitObservations = set()

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
    # print(mu)
    R_robot2base = R.from_euler("z",theta,degrees=False)
    R_base2robot = R_robot2base.inv()
    return R_robot2camera.apply((R_base2robot.apply(mu-position)))

def h_inv(measurement,state):
    theta =state[2]
    R_robot2base=R.from_euler("z",theta,degrees=False)
    position = state[0:2]
    position = np.append(position,0)
    return (R_robot2base.apply(R_robot2camera.inv().apply(np.array(measurement)))+position)[0:2]

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

def isPercievable(beacon_pos : np.ndarray,robot_state : np.ndarray):
    predicted_measurement = h(robot_state,beacon_pos)
    if abs(np.arctan2(predicted_measurement[0],predicted_measurement[2]))>=np.pi/8:
        return False
    if linalg.norm(predicted_measurement)<0.7 or linalg.norm(predicted_measurement)>3:
        return False
    return True

def drawWeight(weightList):
    total_sum = sum(weightList)
    # print("SUM:",total_sum,"\n")
    normalized_weights = [weight / total_sum for weight in weightList]
    draw = random.uniform(0,1)
    previous = 0
    for i in range(len(normalized_weights)):
        if draw >= previous and draw < (previous + normalized_weights[i]):
            return i
        else:
            previous += normalized_weights[i]

    

def systematic_resample(weights):
    N = len(weights)
    total_sum = sum(weights)
    weights = [weight / total_sum for weight in weights]
    # make N subdivisions, choose positions 
    # with a consistent random offset
    positions = (np.arange(N) + random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

class Beacon:
    def __init__(self, pos: np.ndarray, cov: np.ndarray):
        self.tau = 0
        self.position = pos
        self.covariance = cov



class Particle:
    def __init__(self,x : np.ndarray):
        self.N=0
        self.x = x

        self.features = []
        self.weights = []

    #Returns feature number i   
    def get_feature(self,i : int):
        return self.features[i]
    
    def add_feature(self, feature: Beacon):
        self.features.append(feature)

    def remove_feature(self, i : int):
        self.features.pop(i)
    def add_weight(self, weight):
        self.weights.append(weight)
    def remove_weight(self, i : int):
        self.weights.pop(i)

    

    
class ParticleSet:
    def __init__(self,M:int):
        self.M=M
        self.set =[]
    #appends a particle to the list
    def add(self,particle:Particle):
        self.set.append(particle)

    def remove(self,i:int):
        return self.set.pop(i)
    
# def w_AvgOfFeatures(particleset : ParticleSet):
#     beaconMeans = [None]*particleset.set[0].K
#     beaconCovs = [None]*particleset.set[0].K
#     for j in range(particleset.set[0].K):
#         for i in range(particleset.M):
#             beaconMeans
        




### MAKING SOME ARTIFICIAL TEST DATA ###

testSet =ParticleSet(400)

for i in range(testSet.M):
    #Make them random
    testSet.add(Particle(np.array([random.random()-0.5,random.random()-0.5,0]).astype(float)))
    # testSet.add(Particle(3,(np.array([0,0,0])).astype(float)))

def generate_beacons(num_beacons, x_range, y_range):
    beacons = np.random.randint(low=0, high=max(x_range, y_range), size=(num_beacons, 2))
    return beacons

def generate_sampling_positions(num_positions, x_range, y_range):
    positions = np.random.randint(low=0, high=max(x_range, y_range), size=(num_positions, 2))
    angles = np.random.uniform(low=0, high=2*np.pi, size=(num_positions, 1))
    sampling_positions = np.concatenate((positions, angles), axis=1)
    return sampling_positions

def compute_inputs(sample_positions, current_position):
    inputs = []
    for i in range(len(sample_positions)):
        delta_state = sample_positions[i] - current_position
        distance = np.sqrt(delta_state[0]**2 + delta_state[1]**2)
        angle = np.arctan2(delta_state[1], delta_state[0]) - current_position[2]
        inputs.append(np.array([distance, angle]))
        inputs.append(np.array([0,sample_positions[i,2]-np.arctan2(delta_state[1], delta_state[0])]))
        current_position = sample_positions[i]
    return inputs

def compute_observations(sample_positions, beacons):
    observations = []
    for i in range(len(sample_positions)):
        observations_i=[]
        for b in range(len(beacons)):
            if isPercievable(beacons[b],sample_positions[i]):
                observation = h(sample_positions[i],beacons[b])
                observations_i.append(observation)
        observations.append(observations_i)
    return observations

#Size of map
x_map = 15
y_map =15
numBeacons = 30
numPositions = 20
# beacons = np.array([[3,2],[5,5],[1,4]])
X_t1 = np.array([0,0,0])
# samplePositions = np.array([[1,1,0],[2,1,np.pi/4],[3,3,np.pi/4],[3,3,np.pi],[3,3,3*np.pi/2]])
# correlations = [0,0,1,2,0]
beacons = generate_beacons(numBeacons,x_map,y_map)
samplePositions = generate_sampling_positions(numPositions,x_map,y_map)
# print(samplePositions)
inputs = compute_inputs(samplePositions,X_t1)
# print(inputs)

observations = compute_observations(samplePositions,beacons)
# print(len(samplePositions))
# inputs =[]
# observations=[]

# for i in range(len(samplePositions)):
    
#     newInput = [None]*2
#     deltaState = samplePositions[i]-X_t1
#     newInput[1]=np.arctan2(deltaState[1],deltaState[0])-X_t1[2]
#     newInput[0]=np.sqrt(pow(deltaState[1],2)+pow(deltaState[0],2))
#     inputs.append(newInput)

#     inputs.append(np.array([0,samplePositions[i,2]-np.arctan2(deltaState[1],deltaState[0])]))
#     X_t1 = samplePositions[i]

# for i in range(len(samplePositions)):
#     observations.append(h(samplePositions[i],beacons[correlations[i]]))



### ###

def predict2(u_t : np.ndarray, set : ParticleSet, k : int, covariances :  np.ndarray):
    #u_t on the form (delta_x,delta_y,delta_theta)
    xvar = covariances[0]
    yvar = covariances[1]
    thetavar = covariances[2]
    cov_matrix = np.array([[xvar,0,0],[0,yvar,0],[0,0,thetavar]])
    sampledInput = np.random.multivariate_normal(u_t,cov_matrix)
    p = set.set[k]

    (p.x)[2] += sampledInput[2]
    (p.x)[0] += sampledInput[0]
    (p.x)[1] += sampledInput[1]




def predict(u_t : np.ndarray, set :ParticleSet, delta_t : float,k :int,covariances: np.ndarray, usingVelocities = False):
    if usingVelocities:
        if(u_t[0]!=0 or u_t[1]!=0):

            xvar = covariances[0]
            thetavar = covariances[1]
            x_theta_cov = covariances[2]
            cov_matrix = np.array([[xvar,x_theta_cov],[x_theta_cov,thetavar]])
            sampledInput = np.random.multivariate_normal(u_t,cov_matrix)
            p = set.set[k]

            (p.x)[2] +=delta_t*sampledInput[1]
            (p.x)[0] +=delta_t*np.cos((p.x)[2])*sampledInput[0]
            (p.x)[1] +=delta_t*np.sin((p.x)[2])*sampledInput[0]
    else:
        xvar = covariances[0]
        yvar = covariances[1]
        thetavar = covariances[2]
        cov_matrix = np.array([[xvar,0,0],[0,yvar,0],[0,0,thetavar]])
        sampledInput = np.random.multivariate_normal(u_t,cov_matrix)
        p = set.set[k]

        (p.x)[2] += sampledInput[2]
        (p.x)[0] += sampledInput[0]
        (p.x)[1] += sampledInput[1]

def predictMidpoint(u_t : np.ndarray, set :ParticleSet, delta_t : float,k :int):
    p = set.set[k]
    theta_mid = (p.x)[2] +delta_t*(u_t[1])*0.5
    (p.x)[2] +=delta_t*(u_t[1])
    (p.x)[0] +=delta_t*np.cos(theta_mid)*u_t[0]*(1+15*(random.random()-0.5)/50)
    (p.x)[1] +=delta_t*np.sin(theta_mid)*u_t[0]*(1+15*(random.random()-0.5)/50)


def FastSLAM(z_t: np.ndarray,u_t : np.ndarray, Y_t1 :ParticleSet):
    yt1 = Y_t1
    weights = []
    
    for k in range(yt1.M):
        measurement = z_t
        measurement =np.array([measurement[0],measurement[2]])
        currentParticle = yt1.set[k]
        #Predict pose
        # predictMidpoint(u_t,yt1,delta_t,k)
        predict(u_t,yt1,1,k,variances,usingVelocities=True)
        measurement_covariances = []
        measurement_predictions = []

        for j in range(currentParticle.N):
            z_jhat = h(currentParticle.x,currentParticle.features[j].position)
            measurement_predictions.append(z_jhat)
            jacobian_j = h_jacobian(currentParticle.x)
            Q_j = np.matmul(np.matmul(jacobian_j,currentParticle.features[j].covariance),np.transpose(jacobian_j))
            Q_j += Q_t
            measurement_covariances.append(Q_j)

            z_jhat =np.array([z_jhat[0],z_jhat[2]])
            
            

            w_j =  np.power(linalg.det(2*np.pi*Q_j),-0.5)*np.exp(-0.5*np.matmul(np.transpose((measurement-z_jhat)),np.matmul(linalg.inv(Q_j),(measurement-z_jhat))))
            currentParticle.weights[j]=w_j

        
        #TODO: set w_new to a proper p_0
        w_new = 1.0
        currentParticle.add_weight(w_new)
        wk = max(currentParticle.weights)
        weights.append(wk)
        c_hat = np.argmax(currentParticle.weights)
        N_t = max(currentParticle.N,c_hat+1)
        # features2bDiscarded = []
        if(currentParticle.N!=c_hat):#If biggest weight is not the new feature, remove it from weights
            currentParticle.remove_weight(currentParticle.N)
        for j in range(N_t):
            if(j==c_hat and j==currentParticle.N):
                mu_j = h_inv(z_t,currentParticle.x)
                H_j = h_jacobian(currentParticle.x)
                cov_j = np.matmul(np.matmul(np.transpose(linalg.inv(H_j)),Q_t),linalg.inv(H_j))
                newFeature = Beacon(mu_j,cov_j)
                newFeature.tau+=1
                hitObservations.add((k,j))
                currentParticle.add_feature(newFeature)
            elif(j==c_hat and j<currentParticle.N):
                H_j = h_jacobian(currentParticle.x)

                K = np.matmul(np.matmul(currentParticle.features[j].covariance,np.transpose(H_j)),linalg.inv(measurement_covariances[j]))

                z_hat =np.array([measurement_predictions[j][0],measurement_predictions[j][2]])
            

                currentParticle.features[j].position += np.matmul(K,(measurement-z_hat))
                currentParticle.features[j].covariance = np.matmul((np.eye(2)-np.matmul(K,H_j)),currentParticle.features[j].covariance)
                currentParticle.features[j].tau +=1
                hitObservations.add((k,j))
            else:
                if  isPercievable(currentParticle.features[j].position,currentParticle.x):
                    missedObservations.add((k,j))
                    # currentParticle.features[j].tau -=1
                    # if(currentParticle.features[j].tau<0):
                    #     features2bDiscarded.append(j)

        # currentParticle.features = list(np.delete(currentParticle.features,features2bDiscarded))
        # currentParticle.weights = list(np.delete(currentParticle.weights,features2bDiscarded))
        currentParticle.N = len(currentParticle.weights)


    

    Yt = ParticleSet(yt1.M)
    drawnIndexes = systematic_resample(weights)

    for i in drawnIndexes:
        # k=drawWeight(weights)
        p =copy.deepcopy(yt1.set[i])
        Yt.add(p)
    
    print("\n")
    return Yt


points = []
points.append(np.array([testSet.set[i].x for i in range(testSet.M)]))
for i in range(len(samplePositions)):
    particlePoints=[]
    for j in range(testSet.M):

        # print(testSet.set[j].x)
    
        predict(inputs[2*i],testSet,1,j,variances,usingVelocities=True)
        # print(testSet.set[j].x)
        particlePoints.append(testSet.set[j].x)

    points.append(np.array(particlePoints))


    particlePoints=[]
    for j in range(len(observations[i])):
        if(j==0):
            newSet = FastSLAM(observations[i][j],inputs[2*i+1],testSet)
        else:
            newSet = FastSLAM(observations[i][j],np.array([0.0,0.0]),newSet)
    if len(observations[i])==0:
        for j in range(testSet.M):
            predict(inputs[2*i+1],testSet,1,j,variances,usingVelocities=True)
        newSet=testSet
        print(i)

    temp=[]
    for m in missedObservations:
        if m in hitObservations:
            temp.append(m)
    for j in temp:
        missedObservations.remove(j)

    for m in missedObservations:
        newSet.set[m[0]].features[m[1]].tau-=1
        if newSet.set[m[0]].features[m[1]].tau <0:
            newSet.set[m[0]].features = list(np.delete(newSet.set[m[0]].features,m[1]))
            newSet.set[m[0]].weights = list(np.delete(newSet.set[m[0]].weights,m[1]))
            newSet.set[m[0]].N = len(newSet.set[m[0]].weights)

    # newSet = FastSLAM(observations[i],inputs[2*i+1],testSet)
    # print(observations[i])
    for j in range(testSet.M):
        particlePoints.append(newSet.set[j].x)

    points.append(np.array(particlePoints))
    # print(particlePoints)
    testSet=newSet


# for i in range(10):


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5,5)
print(len(points))
def animate(i):
    ax.clear()
    # Get the point from the points list at index i
    pointList = points[i]
    # Plot that point using the x and y coordinates
    ax.scatter(pointList[:,0], pointList[:,1], color='green', 
            label='original', marker='.')
    ax.scatter(beacons[:,0],beacons[:,1],color='red',marker='^')
    ax.scatter(samplePositions[int(i/2),0],samplePositions[int(i/2),1],color='b',marker='o')
    # Set the x and y axis to display a fixed range
    ax.set_xlim([-1, x_map+1])
    ax.set_ylim([-1, y_map+1])
ani = FuncAnimation(fig, animate, frames=len(points),
                    interval=1000, repeat=False)
plt.show()
plt.close()


