import numpy as np
import copy as copy
import scipy as sc
from scipy.spatial.transform import Rotation as R
import scipy.linalg as linalg
import random as random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse


####    ALGORTIHM   ####

featureIDs =set()
R_robot2camera = R.from_euler("yx",[90,-90],degrees=True).inv()
Q_t =np.array(0.001*np.eye(2))
variances = np.array([0.01,0.01,0.01])

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
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
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
    


class Beacon:
    def __init__(self, id=None, mean=None, covariance=None):
        self.id = id
        self.mean = mean
        self.covariance = covariance



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

def predict(u_t : np.ndarray, set :ParticleSet, delta_t : float,k :int,covariances: np.ndarray):
    xvar = covariances[0]
    thetavar = covariances[1]
    x_theta_cov = covariances[2]
    cov_matrix = np.array([[xvar,x_theta_cov],[x_theta_cov,thetavar]])
    sampledInput = np.random.multivariate_normal(u_t,cov_matrix)
    p = set.set[k]

    (p.x)[2] +=delta_t*sampledInput[1]
    (p.x)[0] +=delta_t*np.cos((p.x)[2])*sampledInput[0]
    (p.x)[1] +=delta_t*np.sin((p.x)[2])*sampledInput[0]

# inputs=[[1,0],[0,1],[1,1]]

     
def FastSLAM(z_t: np.ndarray, c_t : int ,u_t : np.ndarray, Y_t1 :ParticleSet, delta_t : float, knownBeacons : np.ndarray):
    yt1 = Y_t1
    weights = np.ones((yt1.M))*(1/yt1.M)

    for k in range(yt1.M):
        measurement = z_t
        currentParticle = yt1.set[k]
        #Predict pose
        predict(u_t,yt1,delta_t,k,variances)

        j=c_t

        if j not in featureIDs:

            # Creates new beacon in beacons' list
            knownBeacons[j] = Beacon(id=j)

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
    drawnIndexes = systematic_resample(Yt.weights)

    for i in drawnIndexes:
        p =copy.deepcopy(yt1.set[i])
        Yt.add(p)
    return Yt

########


####    GENERATE ARTIFICIAL DATA    ####

def generate_positions(n, y_expression, x_max):
    # Generate x values
    xs = np.linspace(0, x_max, n)

    # Calculate y values
    ys = y_expression(xs)
      
    # Calculate arctan of the slope between consecutive points
    arctan_vals = np.arctan(np.diff(ys) / np.diff(xs))
    arctan_vals = np.insert(arctan_vals, 0, 0)  # Add 0 for the first point

    # Combine x, y, and arctan arrays
    positions = np.column_stack((xs, ys, arctan_vals))

    return positions


def y_expression (x):
    # return x
    return np.sin(x) + x

def array_of_beacons ():
    return np.array([[0,0], [3,2],[5,5],[1,4]])


def generate_closed_loop_path(radius, center_x, center_y, num_points):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    angles = np.arctan2(np.diff(y), np.diff(x))
    angles = np.append(angles, angles[0])  # Wrap around to include the first angle
    waypoints = np.column_stack((x, y, angles))
    return waypoints

def prepare_inputs(positions): 

    X_t1 = positions[0]

    inputs =[]
    for i in range(len(positions)):
            
        newInput = [None]*2
        deltaState = positions[i]-X_t1
        newInput[1]=np.arctan2(deltaState[1],deltaState[0])-X_t1[2]
        newInput[0]=np.sqrt(pow(deltaState[1],2)+pow(deltaState[0],2))
        inputs.append(newInput)

        inputs.append(np.array([0,positions[i,2]-np.arctan2(deltaState[1],deltaState[0])]))
        X_t1 = positions[i]
          
    return inputs

    #np.savetxt('inputs.txt', positions, delimiter=' ', fmt='%.6f')

def make_correlations(positions, numInputs):
    correlations = np.zeros(numInputs, dtype=int)
    for i in range(numInputs):
        if positions[i][0]<1:
            if i%2 == 0:
                correlations[i] = 0
            else:
                correlations[i] = 1
        elif positions[i][0]>=1 and positions[i][0]<2:
            correlations[i] = 1
        elif positions[i][0]>=2 and positions[i][0]<3:
            if i%2 == 0:
                correlations[i] = 1
            else:
                correlations[i] = 2
        elif positions[i][0]>=3 and positions[i][0]<4:
            correlations[i] = 2
        elif positions[i][0]>=4 and positions[i][0]<7:
            correlations[i] = 3
        else:
            correlations[i] = 0
    
    return correlations

########


#### Functions relevant for the animated visualization of the algorithm
def estimate_position(allParticles: ParticleSet):
    N = len(allParticles.weights)
    total_sum = sum(allParticles.weights)
    if total_sum == 0:
        return None
    allParticles.weights = [(weight / total_sum) for weight in allParticles.weights]
    mean = np.zeros(2)

    for j in range(allParticles.M):
        particle = allParticles.set[j]
        mean += particle.x[:2] * allParticles.weights[j]
    
    return mean

def update_beacon(allParticles: ParticleSet, i: int, knownBeacons: np.ndarray):
    # Normalize the weights
    N = len(allParticles.weights)
    total_sum = sum(allParticles.weights)
    allParticles.weights = [weight / total_sum for weight in allParticles.weights]

    if knownBeacons[i] is not None:
        mean = np.zeros(2)  # Initialize mean as a zero vector
        covariance = np.zeros((2, 2))  # Initialize covariance as a zero matrix

        for j in range(N):
            particle = allParticles.set[j]
            feature = particle.get_feature(i)
            mean += feature["mean"] * allParticles.weights[j]
            covariance += feature["covariance"] * allParticles.weights[j]
        
        # Update
        knownBeacons[i].mean = mean
        knownBeacons[i].covariance = covariance

def plot_ellipse(mean, covariance, ax):
    w, v = np.linalg.eig(40 * covariance)
    width = 2 * np.sqrt(w) * np.sqrt(40)
    angle = np.degrees(np.arctan2(v[1, 0].real, v[0, 0].real))  # Ensure real values are used

    ell = Ellipse(xy=mean, width=width[0], height=width[1], angle=angle, edgecolor='red', facecolor='none')
    ax.scatter(mean[0], mean[1], color='red', marker='^', s=40, linewidths=1.5)

    ax.add_patch(ell)


def rms_error (ground_truth, estimated):
    ground_truth_2d = np.reshape(ground_truth, (1, 2))
    squared_error = np.sum((ground_truth_2d - estimated) ** 2)
    mean_squared_error = np.mean(squared_error)
    rms_error = np.sqrt(mean_squared_error)
    return rms_error

####
  


def main ():

    beacons = np.array([[1,4],[3,2],[5,3],[5,5]])
    positions = generate_positions(100, y_expression, 7)
    #positions = generate_closed_loop_path(2, 5, 4, 100)
    inputs = prepare_inputs(positions)

    xs = []
    ys = []
    for i in positions:
        x = float(i[0])
        y = float(i[1])
        #theta = float(i[2])
        xs.append(x)
        ys.append(y)
        
    positions = np.array(positions)
    xs = np.array(xs)
    ys = np.array(ys)
    numInputs = len(positions)
    
    correlations = make_correlations(positions, len(positions))

    observations = []
    for i in range(numInputs):
        observations.append(h(positions[i],beacons[correlations[i]]))

    ### ADD INITIAL PARTICLE SET AND BEACONS TO THE ENVIRONMENT ####
    knownBeacons = [Beacon() for _ in range(140)] # Maximum number of beacons
    testSet =ParticleSet(200)
    for i in range(testSet.M):
        testSet.add(Particle(len(beacons),np.array([random.random()*2-1,random.random()*2-1,0]).astype(float)))
    #### extra set for future analysis: considering prediction only without updating particles upon measurements
    extraSet =ParticleSet(200)
    for i in range(extraSet.M):
        extraSet.add(Particle(len(beacons),np.array([random.random()*2-1,random.random()*2-1,0]).astype(float)))
    
    ########

    #### RUN THE ALGORITHM ####
    points = []
    points.append(np.array([testSet.set[i].x for i in range(testSet.M)]))
    estimatedPositions = []
    #estimatedPositions.append([0,0])

    odomes = []
    odomes.append(np.array([extraSet.set[i].x for i in range(extraSet.M)]))
    estimatedOdomes = []
    estimatedOdomes.append([0,0])

    allBeacons = [] # For storing the beacons mean and covariance [mean,covariance]

    rmse_values = []
    rmse_values.append(0)
    rmse_values_odom = []
    rmse_values_odom.append(0)
    trace_values=[]
    trace_values.append(0)

    for i in range(numInputs):

        # Store sets of particles and estimated positions based on those
        particlePoints=[]
        odomePoints=[]
        rmse=[]
        rmse_odom=[]
        
        for j in range(testSet.M):
            predict(inputs[2*i],testSet,1,j, variances)
            particlePoints.append(testSet.set[j].x)

            predict(inputs[2*i],extraSet,1,j, variances)
            odomePoints.append(extraSet.set[j].x)

        points.append(np.array(particlePoints))
        estimated_pos = estimate_position(testSet)
        estimatedPositions.append(estimated_pos)

        rmse = rms_error(positions[i][:2], estimated_pos)
        rmse_values.append(rmse)

        odomes.append(np.array(odomePoints))
        estimated_pos_odom = estimate_position(extraSet)
        estimatedOdomes.append(estimated_pos_odom)

        rmse_odom = rms_error(positions[i][:2], estimated_pos_odom)
        rmse_values_odom.append(rmse_odom)

        particlePoints=[]
        rmse=[]
        rmse_odom=[]
        newSet = FastSLAM(observations[i],correlations[i],inputs[2*i+1],testSet,1, knownBeacons)
        for j in range(testSet.M):
            particlePoints.append(newSet.set[j].x)
        
        points.append(np.array(particlePoints))
        testSet=newSet
        estimated_pos = estimate_position(newSet)
        estimatedPositions.append(estimated_pos)

        rmse = rms_error(positions[i][:2], estimated_pos)
        rmse_values.append(rmse)

        odomes.append(np.array(odomePoints))
        estimatedOdomes.append(estimated_pos_odom)

        rmse_odom = rms_error(positions[i][:2], estimated_pos_odom)
        rmse_values_odom.append(rmse_odom)

        # Compute and store the mean and covariance for each beacon for each set of particles
        beaconsParameters=[]
        trace=0
        for j in featureIDs:
            beaconsParameters.append([knownBeacons[j].mean,knownBeacons[j].covariance])
            if j == 2 and knownBeacons[2].covariance is not None and np.any(knownBeacons[2].covariance != None):
                trace = np.trace(knownBeacons[2].covariance)
        allBeacons.append(beaconsParameters)
        trace_values.append(trace)

        beaconsParameters = []
        for j in featureIDs:
            update_beacon(testSet, j, knownBeacons)
            beaconsParameters.append([knownBeacons[j].mean,knownBeacons[j].covariance])
            if j == 2 and knownBeacons[2].covariance is not None and np.any(knownBeacons[2].covariance != None):
                trace = np.trace(knownBeacons[2].covariance)
        allBeacons.append(beaconsParameters)
        trace_values.append(trace)
    
    x_values = np.repeat(xs, 2)

    ########

    ####    ANIMATION   ####

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax2_2 = ax2.twinx()
    fig.set_size_inches(12,5) 

    plt.style.use("fivethirtyeight")

    # Set the x and y axes limits
    plt.xlim(-1, 8)
    plt.ylim(-1, 8)

    # Create an empty line plot
    path, = plt.plot([], [], linewidth=2)

    # Initialize the animation function
    def animate(i):
        if i == 0:
            return
        i = i-1
        ax1.clear()

        ax1.set_xlim([-1, 8])
        ax1.set_ylim([-1, 8])
        ax1.set_title('FastSLAM KC with artificial data', fontsize=12)
        ax1.set_xlabel('x [m]', fontsize=10)
        ax1.set_ylabel('y [m]', fontsize=10)

        # real position of the robot
        x_data = xs[:(i//2)+1]
        y_data = ys[:(i//2)+1]
        ax1.scatter(x_data[i//2], y_data[i//2], color='green', marker='o')
        ax1.plot(x_data, y_data, color='green', label='True Robot Trajectory', linewidth=1)

        # real position of the beacons
        ax1.scatter(beacons[:,0], beacons[:,1], color = "green", label='True Beacon Position', marker='^', s=40, linewidths=1.5)

        # particles
        pointList = points[i]
        ax1.scatter(pointList[:,0], pointList[:,1], color='blue', marker='.', s=10, linewidths=1)
        
        # estimated position of the robot
        estimatedPosition = estimatedPositions[i]
        estimated_x_data = [pos[0] for pos in estimatedPositions[:i]]
        estimated_y_data = [pos[1] for pos in estimatedPositions[:i]]
        ax1.plot(estimated_x_data, estimated_y_data, color='blue', label='Estimated Robot Trajectory', linewidth=1)
        ax1.scatter(estimatedPosition[0], estimatedPosition[1], color='blue', marker='o')
        
        # estimated position of the robot if it only considers motion model, disregarding measurement model
        estimatedOdome = estimatedOdomes[i]
        odome_x_data = [pos[0] for pos in estimatedOdomes[:i]]
        odome_y_data = [pos[1] for pos in estimatedOdomes[:i]]
        ax1.plot(odome_x_data, odome_y_data, color='black', label='Odometry Robot Position', linewidth=1)
        ax1.scatter(estimatedOdome[0], estimatedOdome[1], color='black', marker='o')

        # estimated position of the beacons
        beaconList = allBeacons[i-1]
        for j in range(len(beaconList)):
            if (i)!=0 and beaconList[j][0] is not None and beaconList[j][1] is not None:
                plot_ellipse(beaconList[j][0], beaconList[j][1], ax1)
        ax1.scatter([],[], color='red', label='Estimated Beacon Position', marker='^', s=40, linewidths=1.5)

        ax1.scatter(0,0,color="orange", marker='*', s=80, linewidths=1)
        ax1.legend(loc='upper left', fontsize=7)


        ax2.clear()
        ax2_2.clear()
        ax2.set_xlim([-1, 8])
        ax2.set_ylim([0, np.max(rmse_values_odom)])
        ax2.set_title('Evolution of RMSE and Trace of Covariance Matrix', fontsize=12)
        ax2.set_xlabel('x [m]', fontsize=10)
        ax2.set_ylabel('RMSE', fontsize=10)

        ax2_2.set_ylim([0, 1.1*np.max(trace_values)])
        ax2_2.set_ylabel('Trace Values', fontsize=10)
        ax2_2.yaxis.set_label_coords(1.25, 0.5)

        # Calculate the root mean squared error error between ground truth and estimated positions
        ax2.plot(x_values[:i+1], rmse_values[:i + 1], color='blue', label='RMSE between True and Estimated Position', linewidth=1)
        ax2.plot(x_values[:i+1], rmse_values_odom[:i + 1], color='black', label='RMSE between True and Odometry Position', linewidth=1)
        ax2_2.plot(x_values[:i+1], trace_values[:i+1], color='red', linewidth=1)
        ax2.plot([],[], color='red', label='Trace of covariance matrix of beacon [5,3]', linewidth=1)
        

    
        ax2.legend(loc='upper left', fontsize=7)
        
        return path,

    # Create the animation
    animation = FuncAnimation(plt.gcf(), animate, frames=len(points), interval=100, blit=False, repeat=True)
    # Show the plot
    plt.show()


    ########


if __name__== '__main__':
    main()
