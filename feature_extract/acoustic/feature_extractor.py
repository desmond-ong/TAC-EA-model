'''
    Combining all the multimodal processing into a single featureExtractor class.

	Acoustic features using opensmile 2.3.0

'''

# other requirements: ffmpeg, opensmile

# directory where opensmile is installed
OPENSMILE_DIRECTORY = "/Users/desmond/Documents/opensmile-2.3.0/"
# directory where the input wav files are
WAV_DIRECTORY = "/Users/desmond/Dropbox/Multimodal/Zen_EA/data/audioWavs"


# I also use a modified emobase_mod.conf configuration file which I modified to output as csv instead of arff

import subprocess
import csv
import os
import scipy
import numpy
import scipy.io.wavfile

class featureExtractor:
    # making a class so we don't have to pickle.load everything all the time.
    def __init__(self, useVisual=True, useAudio=True, useLinguistic=True):
        self.useVisual = useVisual
        self.useAudio = useAudio
        self.useLinguistic = useLinguistic

        #self.features = getFeatureNames(useVisual=useVisual, useAudio=useAudio, useLinguistic=useLinguistic)


    def calculateAcousticFeatures(self, signal, sampleRate, outputFeatureFilename = "features/acoustic/testFile.csv",
    	windowSizeInSeconds=0.5):
        # temporary files. just make sure the opensmile/tmp directory exists
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
            command = OPENSMILE_DIRECTORY + 'SMILExtract -C opensmile/config/emobase_mod.conf -I ' + tempWAVFilename + ' -O ' + tempOutputFilename
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

    def process(self, videoName, windowSizeInSeconds=0.5):
        '''
        audioProcessing.process(videoName)
            Extracts audio features from video

        @input:
            videoName            : (string) a path to the video file
                                    this file only does audioextraction, and assumes a .wav file

        @return:
            none. writes audioFeatureVectors, a [ time x features ] numpy matrix, to csv


        Parameters:
            Default values used for windowSizeInSeconds is 0.5s.

        '''

        baseFilename = os.path.basename(videoName)
        outputFeatureFilename = "features/acoustic/" + baseFilename[:-4] + "_acousticFeatures.csv"

        wavFilename = videoName[:-4] + ".wav"
        # if it doesn't exist, you can call self.convertMP4() on the original mp4 file

        signal, sampleRate = self.readInWAVFile(wavFilename)
        audioFeatureVectors = self.calculateAcousticFeatures(signal=signal, sampleRate=sampleRate,
        	windowSizeInSeconds = windowSizeInSeconds,
        	outputFeatureFilename = outputFeatureFilename)


    def EXTRACT_ALL_AUDIO(self, wav_dir=WAV_DIRECTORY):
    	# apparantly this only takes ~ 1 hr to run through all the wav files, for emo_base configuration.
        for thisFilename in os.listdir(wav_dir):
            if thisFilename.endswith(".wav"):
            	print("EXTRACT_ALL_AUDIO: Processing " + thisFilename)
            	self.process(os.path.join(wav_dir, thisFilename))



# usage:
thisFeatureExtractor = featureExtractor()
#thisFeatureExtractor.EXTRACT_ALL_AUDIO( "PATH_TO_WHERE_WAV_FILES_ARE ")

