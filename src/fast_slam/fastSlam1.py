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
Q_t =np.array(0.0025*np.eye(2))

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
        try:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        except:
            print(weights)
            break
    return indexes



class Particle:
    def __init__(self, K: int,x : np.ndarray):
        self.K=K
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
        self.weights = np.ones((self.M))*(1/self.M)
    #appends a particle to the list
    def add(self,particle:Particle):
        self.set.append(particle)

    def remove(self,i:int):
        return self.set.pop(i)
    

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
        if(u_t[0]!=0 or u_t[1]!=0 or u_t[2]!=0):
            xvar = covariances[0]
            yvar = covariances[1]
            thetavar = covariances[2]
            cov_matrix = np.array([[xvar,0,0],[0,yvar,0],[0,0,thetavar]])
            sampledInput = np.random.multivariate_normal(u_t,cov_matrix)
            p = set.set[k]

            (p.x)[2] += sampledInput[2]
            (p.x)[0] += sampledInput[0]
            (p.x)[1] += sampledInput[1]

  
def FastSLAM(z_t: np.ndarray, c_t : int ,u_t : np.ndarray, Y_t1 :ParticleSet, delta_t : float, usingVelocities = False):
    yt1 = Y_t1
    weights = np.ones((yt1.M))*(1/yt1.M)
    
    for k in range(yt1.M):
        measurement = z_t
        currentParticle = yt1.set[k]
        #Predict pose
        predict(u_t, yt1, delta_t, k, usingVelocities)

        j=c_t
        
        if j not in featureIDs:
            currentParticle.features[j]["mean"] = h_inv(measurement,currentParticle.x)
            jacobian = h_jacobian(currentParticle.x)
            currentParticle.features[j]["covariance"] = np.matmul(np.matmul(linalg.inv(jacobian),Q_t),np.transpose(linalg.inv(jacobian)))
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

    featureIDs.add(j)
    Yt = ParticleSet(yt1.M)
    Yt.weights = weights
    drawnIndexes = systematic_resample(weights)
    for i in drawnIndexes:
        p =copy.deepcopy(yt1.set[i])
        Yt.add(p)
    print("\n")
    return Yt





def main():
    ### MAKING SOME ARTIFICIAL TEST DATA ###

    testSet =ParticleSet(200)

    for i in range(testSet.M):
        #Make them random
        testSet.add(Particle(3,5*np.array([random.random(),random.random(),0]).astype(float)))
        # testSet.add(Particle(3,(np.array([0,0,0])).astype(float)))

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
    # print(beacons[1])
    # print(samplePositions[2])
    # print(h(samplePositions[2],beacons[1]))
    # for i in range(10):
    #     print(inputs[i])

    points = []
    points.append(np.array([testSet.set[i].x for i in range(testSet.M)]))
    for i in range(5):
        particlePoints=[]
        for j in range(testSet.M):

            # print(testSet.set[j].x)
            predict(inputs[2*i],testSet,1,j)
            # print(testSet.set[j].x)
            particlePoints.append(testSet.set[j].x)

        points.append(np.array(particlePoints))
        particlePoints=[]
        newSet = FastSLAM(observations[i],correlations[i],inputs[2*i+1],testSet,1)
        # print(observations[i])
        for j in range(testSet.M):
            particlePoints.append(newSet.set[j].x)
        points.append(np.array(particlePoints))
        # print(particlePoints)
        testSet=newSet


    # for i in range(10):


    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5,5)

    def animate(i):
        ax.clear()
        # Get the point from the points list at index i
        pointList = points[i]
        # Plot that point using the x and y coordinates
        ax.scatter(pointList[:,0], pointList[:,1], color='green', 
                label='original', marker='.')
        ax.scatter(beacons[:,0],beacons[:,1],color='red',marker='^')
        # Set the x and y axis to display a fixed range
        ax.set_xlim([-1, 8])
        ax.set_ylim([-1, 8])
    ani = FuncAnimation(fig, animate, frames=len(points),
                        interval=500, repeat=False)
    plt.show()
    plt.close()


if __name__== '__main__':
    main()