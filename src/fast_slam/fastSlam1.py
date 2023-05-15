import numpy as np
import scipy as sc
from scipy.spatial.transform import Rotation as R
import scipy.linalg as linalg
import random as random
#This may have to be changed to actual arucoIDs
featureIDs ={}
R_robot2camera = R.from_euler("yx",[90,-90],degrees=True).inv()
Q_t =np.array(0.1*np.eye(2))

#Observation model
def h(state,mu):
    theta =state[2]
    position = state[0:2]
    position = np.array([position,0])
    #Make sure z coordinate is in feature mean
    if len(mu)==2:
        mu = np.array([mu,0])
    else:
        mu=np.array(mu)
    R_robot2base = R.from_euler("z",theta,degrees=False)
    R_base2robot = R_robot2base.inv()
    return R_robot2camera.apply((R_base2robot.apply(mu)-position))

def h_inv(measurement,state):
    theta =state[2]
    R_robot2base=R.from_euler("z",theta,degrees=False)
    position = state[0:2]
    position = np.array([position,0])
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


def drawWeight(weightList : list):
    for i in weightList:
        weightList[i]/=sum(weightList)
    draw =random.uniform(0,1)
    previous =0
    for i in range(len(weightList)):
        if(draw>=previous and draw<(previous+weightList[i])):
            return i
        else:
            previous+=weightList[i]



class Particle:
    def __init__(self, K: int,x : np.ndarray):
        self.K=K
        self.x = x

        self.features = []
        for i in range(K):
            self.features[i]={"mean" : np.array([0,0]), "covariance": np.array(np.eye(2))}

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
    

        # for i in range(M):
        #     self.set.append(Particle(K,np.array([0,0])))

#TODO Currently the particles assumes that feature with identifier "k" is at index k
#TODO Fix this^ such that the right feature is acessed without being in index k
#TODO Update step, requires a definition of the observation model h(x,mu)        
def FastSLAM(z_t: np.ndarray, c_t : int ,u_t : np.ndarray, Y_t1 :ParticleSet, delta_t : float):
    yt1 = Y_t1
    weights = np.array(np.ones()*(1/yt1.M))
    for k in range(yt1.M):
        currentParticle = yt1.set[k]
        #Predict pose
        (currentParticle.x)[0] +=delta_t*np.cos((currentParticle.x)[2])*u_t[0]
        (currentParticle.x)[1] +=delta_t*np.sin((currentParticle.x)[2])*u_t[0]
        (currentParticle.x)[2] +=delta_t*u_t[1]

        j=c_t
        
        if j not in featureIDs:
            currentParticle.features[j]["mean"] = h_inv(z_t,currentParticle.x)
            jacobian = h_jacobian(currentParticle.x)
            currentParticle.features[j]["covariance"] = np.matmul(np.matmul(linalg.inv(jacobian),Q_t),np.transpose(linalg.inv(jacobian)))
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
            currentParticle.features[j]["mean"] += np.matmul(K,(z_t-z_hat))
            #Update covariance:
            currentParticle.features[j]["covariance"] = np.matmul((np.eye(2)-np.matmul(K,jacobian)),currentParticle.features[j]["covariance"])
            #update weight/importance factor:
            weights[k] = np.power(linalg.det(2*np.pi*Q),-0.5)*np.exp(-0.5*np.matmul(np.transpose((z_t-z_hat)),np.matmul(linalg.inv(Q),(z_t-z_hat))))
            



        # for i in featureIDs:
        #     #leave unchanged
        #     print()

    Yt = ParticleSet(yt1.M)
    for i in range(yt1.M):
        k = drawWeight(weights)
        Yt.add(yt1[k])

    return Yt




