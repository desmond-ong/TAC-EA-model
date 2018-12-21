'''
    Combining all the multimodal processing into a single featureExtractor class.


    Features are written to:
    ../features/acoustic
    ../features/linguistic
    ../features/visual

	Acoustic features using opensmile 2.3.0

'''

# other requirements: ffmpeg, opensmile

import global_varnames as GLOBAL

# I put my own local directories in global_varnames.
# currently it has: PICKLED_MODEL_DIRECTORY, OPENSMILE_DIRECTORY, WAV_DIRECTORY


import subprocess
import csv
import os
import scipy
import numpy
import scipy.io.wavfile

class featureExtractor:
    # making a class so we don't have to pickle.load everything all the time.
    def __init__(self, useVisual=False, useAudio=True, useLinguistic=False):
        # visual and linguistic not implemented here yet
        self.useVisual = useVisual
        self.useAudio = useAudio
        self.useLinguistic = useLinguistic

        #self.features = getFeatureNames(useVisual=useVisual, useAudio=useAudio, useLinguistic=useLinguistic)

        if self.useLinguistic:
            self.blPosWordList = pickle.load(open(GLOBAL.PICKLED_MODEL_DIRECTORY + "bingliu_posWordList.p", "r"))
            self.blNegWordList = pickle.load(open(GLOBAL.PICKLED_MODEL_DIRECTORY + "bingliu_negWordList.p", "r"))
            self.blDictionary = dict.fromkeys(self.blPosWordList, 1)
            self.blDictionary.update(dict.fromkeys(self.blNegWordList, -1))

            self.liwcPosDictionary = pickle.load(open(GLOBAL.PICKLED_MODEL_DIRECTORY + "liwc2007_posWords.p", "r"))
            self.liwcNegDictionary = pickle.load(open(GLOBAL.PICKLED_MODEL_DIRECTORY + "liwc2007_negWords.p", "r"))
            self.warrinerValenceDictionary = pickle.load(open(GLOBAL.PICKLED_MODEL_DIRECTORY + "warriner_valence.p", "r"))
            self.warrinerArousalDictionary = pickle.load(open(GLOBAL.PICKLED_MODEL_DIRECTORY + "warriner_arousal.p", "r"))
            self.anewValenceDictionary = pickle.load(open(GLOBAL.PICKLED_MODEL_DIRECTORY + "anew_meanValence.p", "r"))
            self.anewArousalDictionary = pickle.load(open(GLOBAL.PICKLED_MODEL_DIRECTORY + "anew_meanArousal.p", "r"))

            # because of the high cost of loading the glove feature, it has to be specially turned on here.
            #self.useGlove = True
            self.useGlove = False
            if self.useGlove:
                self.gloveDimension=300  # change this as desired.
                self.glove = pickle.load(open(GLOBAL.PICKLED_MODEL_DIRECTORY + "glove" + str(self.gloveDimension) + ".p", "r"))
                self.features.extend(['glove' + str(x) for x in xrange(self.gloveDimension)])

    def calculateLinguisticFeatures(self):
        # todo: finish me
        return


    def calculateAcousticFeatures(self, signal, sampleRate, outputFeatureFilename = "features/acoustic/testFile.csv",
    	windowSizeInSeconds=0.5):
        tempWAVFilename = "opensmile/tmp/tempWAVFilename.wav"
        tempOutputFilename = "opensmile/tmp/tempFeaturesOutput.csv"

        ##loglevelFlag = "quiet"
        ##command_ffmpeg = "ffmpeg -i " + inputFilename + " -ac 1 -vn " + loglevelFlag + outputFilename

        windowWidth = (int)(windowSizeInSeconds * sampleRate)  # windowSize seconds * frameRate samples/second
        signalLength = len(signal)
        numWindows = (int)(signalLength/windowWidth)  # this will be a truncated int
        for j in xrange(numWindows):
            print("audioProcessing: Processing Window Number " + str(j+1) + " out of " + str(numWindows))
            sample = signal[(j*windowWidth):((j+1)*windowWidth)]
            scipy.io.wavfile.write(filename=tempWAVFilename, rate=sampleRate, data=sample)
            command = GLOBAL.OPENSMILE_DIRECTORY + 'SMILExtract -C opensmile/config/emobase_mod.conf -I ' + tempWAVFilename + ' -O ' + tempOutputFilename
            subprocess.call(command, shell=True)
            with open(tempOutputFilename, 'rb') as inputFile:
                thisReader = csv.reader(inputFile, delimiter = ",")
                rowNum = 0
                for thisRow in thisReader:
                    if j==0 and rowNum==0:
                        with open(outputFeatureFilename,'w') as outputReader:
                            outputReader.write(', '.join(thisRow)) # write header row to outputFeatureFilename
                            outputReader.write("\n")
                    if rowNum > 0:
                        thisRow[0] = str(int(thisRow[0]) + j) # increment frameIndex field by j
                        thisRow[1] = str(float(thisRow[1]) + j*windowSizeInSeconds) # increment frameTime field by j*windowSizeInSeconds
                        with open(outputFeatureFilename,'a') as outputReader:
                            outputReader.write(', '.join(thisRow)) # write feature row to outputFeatureFilename
                            outputReader.write("\n")
                    rowNum+=1
            # todo: is there a better way than all these in/out?
            # still have to play around with the configuration file
        return

    def readInWAVFile(self, filename, convertToMono=True):
        '''
        processes an audio file in .WAV format to a numpy matrix

        @input:
            filename    : (string) path of input .wav file.

        @return:
            sampleRate    : (int)    .wav file sampling rate, in Hz
            signal        : the raw amplitude data
        '''
        try:
            [sampleRate, signal] = scipy.io.wavfile.read(filename)
        except IOError:
            print "Error: file not found or other I/O error."
            return (-1, -1)
        if convertToMono:
            if signal.ndim == 2:
                # if signal is stereo, average the two channels
                signal = (signal[:, 0] + signal[:, 1]) / 2.0
        return (signal, sampleRate)

    def convertMP4(self, inputFilename, outputFilename=None, loglevel="quiet", toFormat=".wav"):
        '''
        strips the audio file from MP4, converts to WAV, using ffmpeg (and a subprocess.call)

        @input:
            inputFilename    : (string) path of input .mp4 file.
            outputFilename    : (string; optional) path of output .wav file.
            loglevel         : (string; optional) how verbose do you want the ffmpeg call to be? Default: quiet

        @return:
            None
        '''
        if toFormat not in ["wav", "mp4", "mp3"]:
            print "Error. toFormat not recognized, defaulting to .wav"
            toFormat = "wav"
        if outputFilename is None:
            outputFilename = inputFilename[:-4] + "." + toFormat
        if loglevel not in ["quiet", "fatal", "error", "warning", "info", "verbose", "debug"]:
            print "Error. loglevel not recognized, defaulting to quiet."
            loglevel = "quiet"
        loglevelFlag = "-loglevel " + loglevel + " "  # note, without this flag, info is the default
        try:
            #command = "ffmpeg -i INPUT_FILE.mp3 -ab 160k -ac 2 -vn OUTPUT_FILE.wav"
            #command = "avconv  -i INPUT_FILE.mp3  -ab 160k -ac 2 -vn OUTPUT_FILE.wav"
            # use -ac 1 : only 1 audio channel
            # -vn : no video
            command = "ffmpeg -i " + inputFilename + " -ac 1 -vn " + loglevelFlag + outputFilename
            print("Starting audio extraction operation on " + inputFilename)
            subprocess.call(command, shell=True)
            print("Wrote output .wav file to " + outputFilename)
        except IOError:
            print "Error: file not found or other I/O error."
        return

    def process(self, videoName, transcriptFilename=None, windowSizeInSeconds=0.5):
        '''
        featureExtractor.process(videoName, transcriptFilename)
            Extracts audio features from video

        @input:
            videoName            : (string) a path to the (assumed .mp4) video file

        @return:
            None

            audioFeatureVectors are written to a csv    : (numpy.array) a [ time x features ] numpy matrix


        Parameters:
            Default values used for windowSizeInSeconds is 0.5s.

        '''


        if self.useAcoustic:
            # expecting a .wav in the same folder as the videoName. (Maybe not a good assumption...)
            wavFilename = videoName[:-4] + ".wav"
            if(not os.path.exists(wavFilename)):
                print "The .wav file doesn't exist yet; creating it from the .mp4."
                convertMP4(videoName)

            signal, sampleRate = self.readInWAVFile(wavFilename)

            baseFilename = os.path.basename(videoName)
            outputAcousticFeaturesFilename = "features/acoustic/" + baseFilename[:-4] + "_acousticFeatures.csv"

            self.calculateAcousticFeatures(signal=signal, sampleRate=sampleRate,
            	windowSizeInSeconds = windowSizeInSeconds,
            	outputFeatureFilename = outputAcousticFeaturesFilename)

        if self.useLinguistic:
            if transcriptFilename is None:
                print "Error, Transcript required"
            else:
                print "todo, complete me"

        return
        #return audioFeatureVectors


    def EXTRACT_ALL_ACOUSTIC_FEATURES(self, WAV_DIRECTORY=GLOBAL.WAV_DIRECTORY):
    	# apparantly this only takes ~ 1 hr to run through all the wav files, for emo_base configuration.
        for thisFilename in os.listdir(WAV_DIRECTORY):
            if thisFilename.endswith(".wav"):
            	print("EXTRACT_ALL_ACOUSTIC_FEATURES: Processing " + thisFilename)
            	self.process(os.path.join(WAV_DIRECTORY, thisFilename))





# def calculateFeaturesFromJSON(jsonFilename, outputFilename, myFeatureExtractor, windowSize=5):
#     '''
#     usage:
#     e.g.
#     calculateFeaturesFromJSON('videoProcessing/testOutput/902_vid1_forcedAligned.json',
#         '902/ID902_vid1_words_within_window.tsv')
#     '''
#     json_data = json.loads(open(jsonFilename).read())

#     #myFeatureExtractor = linguisticFeatureExtractor()

#     currentStartTime = 0
#     featureNames = myFeatureExtractor.features
#     tsv = 'time\twords\t' + '\t'.join(featureNames) + '\n'
#     thisString = ''
#     for thisWord in json_data['words']:
#         if thisWord['start'] > currentStartTime:
#             if thisWord['word'][0]!='{':
#                 thisString += thisWord['word'] + " "  # only include non special chars: {}
#         if thisWord['end'] > (currentStartTime + windowSize):
#             textFeatures = myFeatureExtractor.calculateFeaturesFromText(thisString)
#             tsv += str(currentStartTime) + "\t" + thisString + "\t" + '\t'.join([str(x) for x in textFeatures]) + "\n"
#             currentStartTime += windowSize
#             thisString = ''
#     parsed = open(outputFilename, 'w')
#     parsed.write(tsv)
#     parsed.close()
