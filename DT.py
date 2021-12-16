import math
import random
import statistics
import numpy as np
import pandas as pd


# class tree
class Tree:
   def __init__(self, parent=None):
      self.parent = parent
      self.children = []
      self.label = None
      self.classCounts = None
      self.splitFeatureValue = None
      self.splitFeature = None

# calculate the probability
def dataToDistribution(data):
   allLabels = [label for (point, label) in data]
   numEntries = len(allLabels)
   possibleLabels = set(allLabels)

   dist = []
   for aLabel in possibleLabels:
      dist.append(float(allLabels.count(aLabel)) / numEntries)

   return dist

# calculate entropy
def entropy(dist):
   return -sum([p * math.log(p, 2) for p in dist])


def splitData(data, featureIndex):
   # get possible values of the given feature
   attrValues = [point[featureIndex] for (point, label) in data]

   for aValue in set(attrValues):
      # compute the piece of the split corresponding to the chosen value
      dataSubset = [(point, label) for (point, label) in data
                    if point[featureIndex] == aValue]

      yield dataSubset

#calculate information gain
def calculateInformationGain(data, featureIndex):
   entropyGain = entropy(dataToDistribution(data))

   for dataSubset in splitData(data, featureIndex):
      entropyGain -= entropy(dataToDistribution(dataSubset))

   return entropyGain


def homogeneous(data):
    # Return True if the data have the same label
   return len(set([label for (point, label) in data])) <= 1


def majorityVote(data, node):
   ''' Label node with the majority of the class labels in the given data set. '''
   labels = [label for (pt, label) in data]
   choice = max(set(labels), key=labels.count)
   node.label = choice
   node.classCounts = dict([(label, labels.count(label)) for label in set(labels)])

   return node

def buildDecisionTree(data, root, remainingFeatures):
   if homogeneous(data):
      root.label = data[0][1]
      root.classCounts = {root.label: len(data)}
      return root

   if len(remainingFeatures) == 0:
      return majorityVote(data, root)

   # find the index of the best feature to split on
   bestFeature = max(remainingFeatures, key=lambda index: calculateInformationGain(data, index))

   if calculateInformationGain(data, bestFeature) == 0:
      return majorityVote(data, root)

   root.splitFeature = bestFeature

   # add child nodes and process recursively
   for dataSubset in splitData(data, bestFeature):
      aChild = Tree(parent=root)
      aChild.splitFeatureValue = dataSubset[0][0][bestFeature]
      root.children.append(aChild)
      buildDecisionTree(dataSubset, aChild, remainingFeatures - set([bestFeature]))


   return root

#inialize parameters and it will return tree
def decisionTree(data):
   return buildDecisionTree(data, Tree(), set(range(len(data[0][0]))))

def predictePoliticalParty(tree, point):
   if tree.children == []:
      return tree.label
   else:
      matchingChildren = [child for child in tree.children
         if child.splitFeatureValue == point[tree.splitFeature]]
      return predictePoliticalParty(matchingChildren[0], point)

def calculateAccuracy(testData,predictedLabels):
   actualLabels = [label for point, label in testData]
   correctLabels = [(1 if a == b else 0) for a,b in zip(actualLabels, predictedLabels)]
   return sum(correctLabels) / len(actualLabels)

def calculate_size(tree):
    if tree is None:
        return 0
    else:
        if tree.label is not None:
            return calculate_size(tree.children)+1


def train_test_split(data, train_size):
    if isinstance(train_size, float):
        train_size = round(train_size * len(data))

    indices = data.index.tolist()
    train_indices = random.sample(population=indices, k=train_size)

    train_df = data.loc[train_indices]
    test_df = data.drop(train_indices)

    return train_df, test_df
def predicte_missing_values(data):
    for col in data.columns:
        predicted_vote = statistics.mode(data[col])
        # print("res", predicted_vote)
        data[col] = np.where((data[col].values == '?'), predicted_vote, data[col].values)
    return data

if __name__ == '__main__':
    # read data from file
    path = "house-votes-84.data.txt"
    df = pd.read_csv(path, header=None,
                       names=['lable', 'issue1', 'issue2', 'issue3', 'issue4', 'issue5', 'issue6', 'issue7', 'issue8',
                              'issue9', 'issue10', 'issue11', 'issue12', 'issue13', 'issue14', 'issue15', 'issue16'])
    # print(data.head(10))
    # predicte missing values
    trainSize = [0.3,0.4,0.5,0.6,0.7]
    iter = 5
    for size in trainSize:
        print("when train size = " , size)
        accuracy =[]
        for j in range(iter):
            modified_data = predicte_missing_values(df)
            # spilit data to train and test sets
            train,test =train_test_split(modified_data,size)
            # store train and test set in files
            train_array = train.to_numpy()
            np.savetxt("train_file.txt", train_array, fmt="%s")
            test_array = test.to_numpy()
            np.savetxt("test_file.txt", test_array, fmt="%s")

            # read train test to applay decision tree on it
            with open('train_file.txt', 'r') as trainFileContent:
                 trainLines = trainFileContent.readlines()
            trainData = [line.strip().split(' ') for line in trainLines]
            # set data in pair format (issues , political party )
            trainData = [(x[1:], x[0]) for x in trainData]
            # print(data)

            tree  = decisionTree(trainData)

            with open('test_file.txt', 'r') as testFileContent:
                 testLines = testFileContent.readlines()
            testData = [line.strip().split(' ') for line in testLines]
            # set data in pair format (issues , political party )
            testData = [(x[1:], x[0]) for x in testData]
            for i in testData:
                predictedPoliticalParty = [predictePoliticalParty(tree, issues) for issues , political in testData]
            # print(predictedPoliticalParty)
            accuracy.append(calculateAccuracy(testData,predictedPoliticalParty))
        # print(accuracy)
        print("Maximum accuracy =",max(accuracy)*100)
        print("Minimum accuracy =",min(accuracy)*100)
        print("Mean accuracy =", (sum(accuracy)/len(accuracy))*100)
        print("*********************************************************")
