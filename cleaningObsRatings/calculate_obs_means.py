import os
import numpy as np

TRAIN_DIRECTORY = "../ratings/Train/observer/"
VALID_DIRECTORY = "../ratings/Valid/observer/"
TEST_DIRECTORY = "../ratings/Test/observer/"

TRAIN_OUT_DIRECTORY = "../ratings/Train/observer_averaged/"
VALID_OUT_DIRECTORY = "../ratings/Valid/observer_averaged/"
TEST_OUT_DIRECTORY = "../ratings/Test/observer_averaged/"


# THIS_DIRECTORY = TRAIN_DIRECTORY
# THIS_OUT_DIRECTORY = TRAIN_OUT_DIRECTORY
# THIS_DIRECTORY = VALID_DIRECTORY
# THIS_OUT_DIRECTORY = VALID_OUT_DIRECTORY
THIS_DIRECTORY = TEST_DIRECTORY
THIS_OUT_DIRECTORY = TEST_OUT_DIRECTORY

for thisFilename in os.listdir(THIS_DIRECTORY):
    if(thisFilename == ".DS_Store"):
        os.remove(THIS_DIRECTORY + ".DS_Store")
    else:
        # results_111_1
        vidID = thisFilename[8:13]
        fileDescriptor = open(THIS_DIRECTORY + thisFilename, 'r')
        lines = fileDescriptor.readlines()
        fileDescriptor.close()
        #header = lines[0].split(',')
        csv_string = 'time,rating\n'

        for thisRow in xrange(1,len(lines)):
            originalRow = lines[thisRow].split(',')[1:]
            currentRow = [float(x.strip('\n')) for x in originalRow]
            csv_string = csv_string + lines[thisRow].split(',')[0] + "," + "{:.2f}".format(np.mean(currentRow)) + "\n"
            
        parsedFile = open(THIS_OUT_DIRECTORY + thisFilename, 'w')
        parsedFile.write(''.join(csv_string))
        parsedFile.close()
