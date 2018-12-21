import linguisticProcessing
from linguisticProcessing import linguisticFeatureExtractor
from linguisticProcessing import calculateFeaturesFromJSON
from os import listdir
from os.path import isfile, join

myFeatureExtractor = linguisticFeatureExtractor()
input_dir_transcript="./transcript_txt/aligned_txt/"
txt_files = [f for f in listdir(input_dir_transcript) if isfile(join(input_dir_transcript, f))]
for file in txt_files:
	if file[-1] != 'e':
		inputFile = join(input_dir_transcript, file)
		outputFile = join("./feature_output/", file.split('.')[0] + ".tsv")
		windowSize = 5
		calculateFeaturesFromJSON(inputFile, outputFile, myFeatureExtractor, windowSize)
		print "Success! Feature output written to " + outputFile