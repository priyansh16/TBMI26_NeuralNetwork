import numpy as np

def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    """
    steps 
    - calculate the no of classes.
    - inilize the cM with zeros for the square matrix of dim as no of classes.
    - inilize the cM for each pred  vs true values
    - return cM
    """
    
    no_class = max(np.max(LPred), np.max(LTrue))+1 # aMax no of classes from each lables
    cM = np.zeros((int(no_class), int(no_class)), dtype=int) #initilizing it with zeros.

    #adding entries to matrix corresponsing to classes
    for i in range(len(LPred)):
        cM[int(LPred[i])][int(LTrue[i])] += 1 #-1 due to zero based indexing
    return cM

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """ 

    """
    steps 
    - Ensure the input is np array
    - calculate the number of correct pred
    - acc = correct pred / total pred
    - return acc
    """
    #Ensuring that these are numpy array
    Lpred = np.array(LPred)
    Ltrue = np.array(LTrue)

    no_correct_pred = np.sum(Ltrue == Lpred)
    acc = None

    #acc = no of correct prediction / total no data
    acc = no_correct_pred / len(Ltrue)

    return acc




def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """
    """
    steps
    - ensuring the matrix is a np matrix
    - calculate the no of correct predistions(sum of the diagonal elements of cM)
    - calculate the total no of prediction (Sum of each element of cM)
    - acc = correct prediction / total prediction
    - return acc
    """
    #ensuring its a numpy matrix
    cM = np.array(cM)
    no_correct_pred = np.sum(np.diag(cM))
    total_pred = np.sum(cM)
    acc = None
    acc = no_correct_pred / total_pred
    
    return acc
