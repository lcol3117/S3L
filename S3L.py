import functools as ft, random as rnd, math as math
class S3L_agent:
  def __init__(self, dims, mip, j):
    #Class Constructor
    self.dims = dims #Dimensions of Policy Space
    self.mip = mip #Maximum Impossible Performance
    self.j = j #Lone hyperparameter, increase to make model less shy
    self.qtable = [] #Stores the Q-Table with elements [policy, performance]
    self.epsilon = None #Epsilon has not yet been calculated, use calcEpsilon
  def updateQTable(self, policy, performance):
    #Update Q Table with results of policy experimentation
    nextentry = [policy, performance] #To be added to the Q-Table
    self.qtable.append(nextentry) #Update the Q-Table
  def selectPolicy(self):
    #Select a policy as a normalized self.dims-dimentional vector
    self.calcEpsilon() #Update self.epsilon
    try:
      initialpoint = self.generateAvoidancePoint() #See generateAvoidancePoint
    except IndexError:
      initialpoint = list(map(
        lambda _ : rnd.random(),
        range(self.dims)
      )) #Choose a random point if unable to calculate avoidancePoint
    probabilisticmax = self.locateProbabilisticMax() #See locateProbabilisticMax
    vectordelta = self.subVectors(initialpoint, probabilisticmax) #Subtraction
    scalar = self.calcVectorDeltaScalar() #See calcVectorDeltaScalar
    modifiedvectordelta = self.scalarVectorMultiply(vectordelta, scalar) #Multiply
    result = self.vectorSum(initialpoint, modifiedvectordelta) #Add vectors
    return result
  def calcEpsilon(self):
    #Calculate Epsilon as the certainty of extrema
    try: 
      qtablemax = ft.reduce(
        lambda a,x : x[1] if x[1] > a else a,
        self.qtable,
        float("-inf")
      ) #Calculate max(Q)
      res = (qtablemax / self.mip) #Initially, use max(Q)/MIP
      if res >= 0.8:
        res = 0.8 + (((5 * (res - 0.8)) ** self.j) / 5) #Avoid overfitting at end
    except IndexError:
      res = 0 #Default to zero if max(Q)/MIP is not calculable
    self.epsilon = res #Update self.epsilon
  def generateAvoidancePoint(self):
    #Get a random point, but try again if it is too close to minima
    initialvector = list(map(
      lambda _ : rnd.random(),
      range(self.dims)
    )) #Calculate a random normalized n-dimensional vector
    iavgqtable = (3 / 4) * self.mip #Ideal Average performance in Q-Table
    possiblemins = list(filter(
      lambda x : x[1] < iavgqtable,
      self.qtable
    )) #Possible minima
    possiblemaxs = list(filter(
      lambda x : x[1] >= iavgqtable,
      self.qtable
    )) #Possible maxima
    [possibleminvectors, possiblemaxvectors] = list(map(
      lambda x : list(map(
        lambda v : v[0],
        x
      )),
      [possiblemins, possiblemaxs]
    )) #Ignore performance, use only vector itself
    closestmin = possibleminvectors[0] #Default to the first option
    bestdist = self.getL2NDist(closestmin, initialvector)
    for i in possibleminvectors:
      dist = self.getL2NDist(i, initialvector)
      if dist < bestdist:
        bestdist = dist #Update bestdist
        closestmin = i[:] #Update closestmin
    closestmax = possiblemaxvectors[0] #Default to the first option
    bestdist = self.getL2NDist(closestmax, initialvector)
    for i in possiblemaxvectors:
      dist = self.getL2NDist(i, initialvector)
      if dist < bestdist:
        bestdist = dist #Update bestdist
        closestmax = i[:] #Update closestmax
    ddelta = self.getL2NDist(closestmin, closestmax) #Distance max to min
    dmin = self.getL2NDist(initialvector, closestmin) #Distance point to min
    r = dmin / ddelta #Ratio of distances, on a scale of 0 to 1
    if r < 0.5:
      tryagain = rnd.random() <= (r * self.epsilon) #Retry if close to minima
    else:
      tryagain = False #Deadband
    if tryagain:
      return self.generateAvoidancePoint() #Try again if we must
    else:
      return initialvector #Exit case in terms of recursion
  def locateProbabilisticMax(self):
    #Usually the max, but not always
    e2 = self.epsilon / 2 #Probability of swapping
    ls = self.qtable[:] #Q-Table
    ranking = sorted(ls, key = lambda x : x[1])[:] #Rank by performance
    r = list(reversed(ranking))[:] #Reverse sort so maximum is first
    for i in range(len(ranking)-1):
      if rnd.random() < e2:
        tmp = r[i+1][:] #Temp variable
        r[i+1] = r[i][:] #Update first
        r[i] = tmp[:] #Update second
    newranking = r[:] #Avoid collisions
    newtop = newranking[0][0] #Get policy vector at the top of the new list
    return newtop #Return it
  def calcVectorDeltaScalar(self):
    vectordeltascalar = self.epsilon ** self.j #Calculate the scalar
    return vectordeltascalar #Return it
  def subVectors(self, pointa, pointb):
    #Subtract vectors
    regionpt = pointb[:] #Avoid collsions
    r = list(range(self.dims))[:] #Create range object then call __list__
    deltas = list(map(lambda i : (regionpt[i] - pointa[i]), r)) #Subtract
    return deltas #Return the result
  def scalarVectorMultiply(self, oldvector, scalar):
    #Multiply a vector by a scalar
    newvector = list(map(lambda x : x * scalar, oldvector)) #Multiply
    return newvector #Return it
  def vectorSum(self, a, b):
    #Add 2 vectors
    r = list(range(self.dims))[:] #Create range object then call __list__
    vsum = list(map(lambda i : (a[i] + b[i]), r)) #Add them
    return vsum #Return it
  def getL2NDist(self, a, b):
    #Calculate n-dimensional L2 Euclidean distance
    sumd = ft.reduce(
      lambda a,x : a + ((x[0] - x[1]) ** 2),
      list(zip(a, b)),
      0
    ) #Sum is under the square root
    dist = math.sqrt(sumd) #Perform the square root to get the result
    return dist #Return the L2 distance
