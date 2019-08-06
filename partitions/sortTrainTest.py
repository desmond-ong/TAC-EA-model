import os
import shutil

fileDescriptor = open('../TrainSetAssignments/TrainSet.csv', 'r')
linesAll = fileDescriptor.readlines()
fileDescriptor.close()
TrainSet = [vidID.strip('\n') for vidID in linesAll]

fileDescriptor = open('../TrainSetAssignments/ValidSet.csv', 'r')
linesAll = fileDescriptor.readlines()
fileDescriptor.close()
ValidSet = [vidID.strip('\n') for vidID in linesAll]

fileDescriptor = open('../TrainSetAssignments/TestSet.csv', 'r')
linesAll = fileDescriptor.readlines()
fileDescriptor.close()
TestSet = [vidID.strip('\n') for vidID in linesAll]

UNSORTED_DIRECTORY = "../features/Unsorted/linguistic/"
TRAIN_DIRECTORY = "../features/Train/linguistic/"
VALID_DIRECTORY = "../features/Valid/linguistic/"
TEST_DIRECTORY = "../features/Test/linguistic/"

for thisFilename in os.listdir(UNSORTED_DIRECTORY):
    if(thisFilename == ".DS_Store"):
        os.remove(UNSORTED_DIRECTORY + ".DS_Store")
    else:
        vidID = thisFilename[2:6] + thisFilename[9]
        if vidID in TrainSet:
            shutil.move(UNSORTED_DIRECTORY + thisFilename, TRAIN_DIRECTORY + thisFilename)
        elif vidID in ValidSet:
            shutil.move(UNSORTED_DIRECTORY + thisFilename, VALID_DIRECTORY + thisFilename)
        elif vidID in TestSet:
            shutil.move(UNSORTED_DIRECTORY + thisFilename, TEST_DIRECTORY + thisFilename)
        else:
            print "Not found! : " + thisFilename


