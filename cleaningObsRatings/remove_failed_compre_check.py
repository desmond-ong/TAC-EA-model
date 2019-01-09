# raw, without removal, is at:
# "~/Dropbox/Multimodal/shared_code/analysis/processed/"
#
# I used "~/Dropbox/Multimodal/shared_code/analysis/processed/individuals-parsed-nocomments.csv"
# which I reproduce in this folder, to clean the "attention checks"
#
#
#
import os
import shutil

# First, parse the individuals-parsed-nocomments.csv file to get the list of those I want to remove
fileDescriptor = open('individuals-parsed-nocomments.csv', 'r')
linesAll = fileDescriptor.readlines()
fileDescriptor.close()
lines = linesAll[0].split('\r')
header = lines[0].split(',')

NUM_TRIALS = 8
sequenceIndicesStart = header.index('sequence.1')
comprehensionCheckQ1CorrectIndicesStart = header.index('comprehensionCheckQ1Correct.1')
comprehensionCheckQ2CorrectIndicesStart = header.index('comprehensionCheckQ2Correct.1')

sequenceIndices = range(sequenceIndicesStart, sequenceIndicesStart+NUM_TRIALS)
comprehensionCheckQ1CorrectIndices = range(comprehensionCheckQ1CorrectIndicesStart, comprehensionCheckQ1CorrectIndicesStart+NUM_TRIALS)
comprehensionCheckQ2CorrectIndices = range(comprehensionCheckQ2CorrectIndicesStart, comprehensionCheckQ2CorrectIndicesStart+NUM_TRIALS)

# whichIndices = range(whichIndicesStart, (whichIndicesEnd+1))

ExclusionList = {}

for rowNum in xrange(1,len(lines)):
    thisRow = lines[rowNum].split(',')
    thisParticipantID = thisRow[0]
    q1Responses = [thisRow[ind]=='TRUE' for ind in comprehensionCheckQ1CorrectIndices]
    q2Responses = [thisRow[ind]=='TRUE' for ind in comprehensionCheckQ2CorrectIndices]
    # EXCLUDING if they failed EITHER question; they need to get both correct, hence "and"
    bothResponses = [q1Responses[ind] and q2Responses[ind] for ind in xrange(len(q1Responses))]
    for trial in xrange(NUM_TRIALS):
        if bothResponses[trial] is False:
            if thisParticipantID not in ExclusionList:
                ExclusionList[thisParticipantID] = [thisRow[sequenceIndicesStart + trial]]
            else:
                newExclList = ExclusionList[thisParticipantID]
                newExclList.append(thisRow[sequenceIndicesStart + trial])
                ExclusionList[thisParticipantID] = newExclList

exclusionCount = 0
for thisParticipantID in ExclusionList:
    exclusionCount += len(ExclusionList[thisParticipantID])
##993
print "Exclusion Count: " + str(exclusionCount)


RAW_DIRECTORY = "raw/"
UNSORTED_DIRECTORY = "clean/"

for thisFilename in os.listdir(RAW_DIRECTORY):
    if(thisFilename == ".DS_Store"):
        os.remove(RAW_DIRECTORY + ".DS_Store")
    else:
        # results_111_1
        vidID = thisFilename[8:13]
        fileDescriptor = open(RAW_DIRECTORY + thisFilename, 'r')
        lines = fileDescriptor.readlines()
        fileDescriptor.close()
        header = lines[0].split(',')
        ColumnsToExclude = []
        for thisCol in xrange(1, len(header)):
            if header[thisCol] in ExclusionList:
                if vidID in ExclusionList[header[thisCol]]:
                    ColumnsToExclude.append(thisCol)

        newLines = list(lines)
        for thisRow in xrange(len(lines)):
            originalRow = lines[thisRow].split(',')
            currentRow = [originalRow[ind] for ind in xrange(len(originalRow)) if ind not in ColumnsToExclude]
            currentRowString = ','.join(currentRow)
            if (len(originalRow)-1) in ColumnsToExclude:
                currentRowString += '\n'
            newLines[thisRow] = currentRowString

        parsedFile = open(UNSORTED_DIRECTORY + thisFilename, 'w')
        parsedFile.write(''.join(newLines))
        parsedFile.close()


#UNSORTED_DIRECTORY = "../ratings/Unsorted/observer/"

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

TRAIN_DIRECTORY = "../ratings/Train/observer/"
VALID_DIRECTORY = "../ratings/Valid/observer/"
TEST_DIRECTORY = "../ratings/Test/observer/"

for thisFilename in os.listdir(UNSORTED_DIRECTORY):
    if(thisFilename == ".DS_Store"):
        os.remove(UNSORTED_DIRECTORY + ".DS_Store")
    else:
        vidID = thisFilename[8:13]
        if vidID in TrainSet:
            shutil.move(UNSORTED_DIRECTORY + thisFilename, TRAIN_DIRECTORY + thisFilename)
        elif vidID in ValidSet:
            shutil.move(UNSORTED_DIRECTORY + thisFilename, VALID_DIRECTORY + thisFilename)
        elif vidID in TestSet:
            shutil.move(UNSORTED_DIRECTORY + thisFilename, TEST_DIRECTORY + thisFilename)
        else:
            print "Not found! : " + thisFilename


