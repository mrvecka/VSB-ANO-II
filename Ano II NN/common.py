import numpy as np

def compute_accuracy(predicted, ground_truth_path):
    
    ground = []
    with open(ground_truth_path, 'r') as infile_label:

        for line in infile_label:
            line = line.rstrip('\n')
            ground.append(int(line))
           
    # first aproach
    falsePositive = 0
    falseNegative = 0
    truePositive = 0
    trueNegative = 0
    
    for i in range(len(predicted)):
        if (predicted[i] == 1 and ground[i] == 0):
            falsePositive +=1
            
        if (predicted[i] == 0 and ground[i] == 1):
            falseNegative +=1
            
        if (predicted[i] == 1 and ground[i] == 1):
            truePositive +=1
            
        if (predicted[i] == 0 and ground[i] == 0):
            trueNegative +=1  
        
    accuracy_1 = (float(truePositive + trueNegative) / float(truePositive + trueNegative + falsePositive + falseNegative))*100
    print("Accuracy according to teacher: ", accuracy_1)

    # second aproach
    equals = [] 
    for i in range(0,len(ground)):
        if(ground[i] == predicted[i]):
            equals.append(1)
        else:
            equals.append(0)
            
    non_zero = np.count_nonzero(equals)
    accuracy_2 = non_zero / len(equals)
    return accuracy_2 *100