'''
    Header file / Table of Functions for external functions to call:

    by default, put the alignedTranscript file in the same dir as the transcript file itself

    linguisticProcessing.getFeatureNames():
        outputs a list of the feature names, in order.
        @input:
            None
        @output:
            featureNames     : (list) a list of feature names

    linguisticProcessing.process(transcriptFilename, wavfile):
        Extracts linguistic features.
        @input:
            transcriptFilename   : (string) a path to the (.txt) plaintext transcript
            wavfile              : (string) a path to the (.wav) audio source
        @return:
            lingFeatureVectors    : (numpy.array) a [ time x features ] numpy matrix

        Here's the workflow for linguisticProcessing.process(transcriptFilename, wavFilename):
        1) convert the .txt transcript into a jsonTranscriptFile,
            using text_to_transcript()
            -- Format of jsonTranscriptFile:

        2) perform forced alignment using the jsonTranscriptFile and the wavFile, writes to jsonAlignedFile,
            using align.do_alignment()
            -- Format of jsonAlignedFile:
            https://libraries.io/github/ucbvislab/p2fa-vislab


        3) Using the jsonAlignedFile and input dictionaries, calculate features for each timewindow,
            using calculateFeaturesFromJSON()
            -- Format of outputFilename:
            time  \t  'words that were said in this window' \t features

    note: the forced alignment step (step 2) will crap out if there is an unknown word
        (a word that's not found in the CMU pronouncing dictionary)
        sometimes this could just be an "error" in the transcription
        for example, "that memory's vivid" = "that memory is vivid".
        I have a function to replace these words... not very scalable...


    References and Acknowledgements:
    Uses p2fa-vislab (https://github.com/ucbvislab/p2fa-vislab) to do forced alignment.

    --- helper functions: ---

    ## see my Research/Software/Sentiment Lexicon folder

    text_to_transcript(text_file, output_file)
        takes in a plaintext transcript file, and converts it into JSON format,
        which align.do_alignment() requires
        @input:
            text_file              : (string) a path to the (.txt) plaintext transcript
            output_file            : (string) a path to the (.json) JSON formatted transcript output
        @return:
            None


    (helper.p2fa-vislab.) align.do_alignment(wavfile, jsonTranscriptFile, jsonOutputFile)
        takes in an audio file, a JSON transcript file, and does forced alignment.
        @input:
            wavfile              : (string) a path to the (.wav) audio source
            jsonTranscriptFile   : (string) a path to the (.json) JSON formatted transcript
            jsonOutputFile       : (string) a path to the (.json) JSON formatted Forced-Aligned transcript
        @return:
            None

    calculateFeaturesFromJSON(jsonFilename, outputFilename, windowSize=5):
        takes in the forced aligned JSON transcript, calculates features for a given window,
        writes the features to outputFilename (.tsv)
        @input:
            jsonFilename     : (string) a path to the (.json) JSON formatted Forced-Aligned transcript
            outputFilename   : (string) a path to the (.tsv) feature output
            windowSize       : (int) length of the feature window (in SECONDS)
        @return:
            None



'''

import numpy
import pickle
import json
import re  # for regular expressions
import sys
sys.path.append('helper/p2fa-vislab')
#import align
#align = __import__('helper.pf2a-vislab.align')
from helper.p2faVislab import align

# import simplejson as json
import os.path

from unidecode import unidecode # for unicode

# import click
import jsonschema


PICKLED_MODEL_DIRECTORY = "/Users/zhengxuanw/SSNL/Zen_EA/linguistic/models/"


'''
### -------
### Externally-callable Functions
### -------
'''


def getFeatureNames():
    '''
    linguisticProcessing.getFeatureNames()
        outputs a list of the feature names, in order.

    @input:
        None

    @output:
        featureNames     : (list) a list of feature names
    '''
    # return ["wordCount", "MeanSentiment_b", "SDSentiment_b",
    #         "sumLIWCPos", "SDLiwc_neg", "sumLIWCNeg", "SDLiwc_neg"
    #         "MeanValence_w", "SDValence_w", "MeanArousal_w", "SDArousal_w",
    #         "MeanValence_a", "SDValence_a", "MeanArousal_a", "SDArousal_a"]
    return ["wordCount", "MeanSentiment_b", #"SDSentiment_b",
            "SumLIWCPos", #"SDLiwc_neg", 
            "SumLIWCNeg", #"SDLiwc_neg"
            "MeanValence_w", #"SDValence_w", 
            "MeanArousal_w", #"SDArousal_w",
            "MeanValence_a", #"SDValence_a", 
            "MeanArousal_a"]#, "SDArousal_a"]


def process(transcriptFilename, wavfile, windowSize=5, myFeatureExtractor=None):
    '''
    extracts linguistic features from transcriptFile
    '''
    try:
        transcriptFile = open(transcriptFilename)
    except IOError:
        print "Error: transcript file not found or other I/O error."

    # linguisticFeatureVectors = numpy.zeros((numWindows, numFeatures))
    # for line in transcriptFile.readlines():
    #     lineValues = calculateAttributesFromText(line)
    #     linguisticFeatureVectors.append(lineValues)
    # if not myFeatureExtractor:
    #     myFeatureExtractor = linguisticFeatureExtractor()

    scrubbedTranscriptFile = transcriptFilename[:-4] + "_scrubbed.txt"
    jsonTranscriptFile = transcriptFilename[:-4] + "_transcript.json"
    jsonAlignedFile = transcriptFilename[:-4] + "_aligned.json"
    outputFilename = transcriptFilename[:-4] + "_aligned.tsv"

    print "Scrubbing timestamps from " + transcriptFilename + " to " + scrubbedTranscriptFile
    removeTimestampsFromTranscripts(transcriptFilename, scrubbedTranscriptFile)

    print "Performing alignment on audio file " + wavfile + " with transcript file " + scrubbedTranscriptFile
    if os.path.isfile(jsonTranscriptFile):
    #if False:
        print "JSON transcript file: " + jsonTranscriptFile + " already exists, using that instead to perform alignment..."
    else:
        print "Converting " + scrubbedTranscriptFile + " to json format: " + jsonTranscriptFile
        text_to_transcript(scrubbedTranscriptFile, jsonTranscriptFile)
        print "Success! Performing alignment..."

    if os.path.isfile(jsonAlignedFile):
    #if False:
        print "JSON Aligned File: " + jsonAlignedFile + " already exists, using that instead to calculate features..."
    else:
        align.do_alignment(wavfile, jsonTranscriptFile, jsonAlignedFile)
        print "Success! Intermediate output written to " + jsonAlignedFile + ". Skip calculating features..."
    # calculateFeaturesFromJSON(jsonAlignedFile, outputFilename, myFeatureExtractor, windowSize)
    # print "Success! Feature output written to " + outputFilename

    #transcriptFile.close()
    #return linguisticFeatureVectors


'''          
### -------
### Helper Functions
### -------
'''


class linguisticFeatureExtractor():
    # making a class so we don't have to pickle.load everything all the time.
    def __init__(self):
        self.features = getFeatureNames()
        self.blPosWordList = pickle.load(open(PICKLED_MODEL_DIRECTORY + "bingliu_posWordList.p", "r"))
        self.blNegWordList = pickle.load(open(PICKLED_MODEL_DIRECTORY + "bingliu_negWordList.p", "r"))
        self.blDictionary = dict.fromkeys(self.blPosWordList, 1)
        self.blDictionary.update(dict.fromkeys(self.blNegWordList, -1))

        self.liwcPosDictionary = pickle.load(open(PICKLED_MODEL_DIRECTORY + "liwc2007_posWords.p", "r"))
        self.liwcNegDictionary = pickle.load(open(PICKLED_MODEL_DIRECTORY + "liwc2007_negWords.p", "r"))
        self.warrinerValenceDictionary = pickle.load(open(PICKLED_MODEL_DIRECTORY + "warriner_valence.p", "r"))
        self.warrinerArousalDictionary = pickle.load(open(PICKLED_MODEL_DIRECTORY + "warriner_arousal.p", "r"))
        self.anewValenceDictionary = pickle.load(open(PICKLED_MODEL_DIRECTORY + "anew_meanValence.p", "r"))
        self.anewArousalDictionary = pickle.load(open(PICKLED_MODEL_DIRECTORY + "anew_meanArousal.p", "r"))


        # because of the high cost of loading the glove feature, it has to be specially turned on here.
        #self.useGlove = True
        self.useGlove = True
        if self.useGlove:
            self.gloveDimension=300  # change this as desired.
            self.glove = pickle.load(open(PICKLED_MODEL_DIRECTORY + "glove" + str(self.gloveDimension) + ".p", "r"))
            self.features.extend(['glove' + str(x) for x in xrange(self.gloveDimension)])

        




    def calculateAttributeOfWordUsingDictionary(self, word, thisDictionary):
        # change this to include wildcards?
        if word in thisDictionary:
            return thisDictionary[word]
        else:
            # check for wildcards
            wildcardKeys = [key for key in thisDictionary.keys() if '*' in key]
            for key in wildcardKeys:
                if key.startswith(word):
                    return thisDictionary[key]
            # if nothing, return nan
            return numpy.nan

    def extractWordVectorFromWords(self, words):
        if not self.useGlove:
            return numpy.nan
        else:
            wordVecs = numpy.array([self.glove[word] for word in words if word in self.glove])
            # wordVecs gives a n by d array, 
            #    where n is number of words and d is dimension of word vector embedding
            # axis = 0 will give a d-dimensional array of the mean of all the words.
            if len(wordVecs) > 0:
                return wordVecs.mean(axis = 0)
            else:
                return numpy.empty(self.gloveDimension) * numpy.nan


    def calculateFeaturesFromText(self, text):
        # TODO calculate ALL features here
        words = [word.lower().strip(',.;:?!-') for word in text.split()]

        #featureNames = self.features()
        # for currentFeature in featureNames:
        #     if model == "Valence":
        #         thisDictionary = pickle.load(open("models/pickledData/warriner_valence.p", "r"))
        #     elif model == "Arousal":
        #         thisDictionary = pickle.load(open("models/pickledData/warriner_arousal.p", "r"))
        #     #else:
        #     # default
        #     #thisDictionary = pickle.load(open("models/pickledData/warriner_valence.p", "r"))
        #     thisValue = [calculateAttributeOfWordUsingDictionary(word, thisDictionary) for word in words]

        bingliuSentimentValues = [self.calculateAttributeOfWordUsingDictionary(word, self.blDictionary) for word in words]
        liwcPosValues = [self.calculateAttributeOfWordUsingDictionary(word, self.liwcPosDictionary) for word in words]
        liwcNegValues = [self.calculateAttributeOfWordUsingDictionary(word, self.liwcNegDictionary) for word in words]
        warrinerValenceValues = [self.calculateAttributeOfWordUsingDictionary(word, self.warrinerValenceDictionary) for word in words]
        warrinerArousalValues = [self.calculateAttributeOfWordUsingDictionary(word, self.warrinerArousalDictionary) for word in words]
        anewValenceValues = [self.calculateAttributeOfWordUsingDictionary(word, self.anewValenceDictionary) for word in words]
        anewArousalValues = [self.calculateAttributeOfWordUsingDictionary(word, self.anewArousalDictionary) for word in words]

        wordCount = len(words)
        if len(bingliuSentimentValues) > 0:
            meanSentiment_b = numpy.nanmean(bingliuSentimentValues)
            # sdSentiment_b = numpy.nanstd(bingliuSentimentValues)
        else:
            meanSentiment_b = ''
            # sdSentiment_b = ''
        if len(liwcPosValues) > 0:
            sumLIWCPos = numpy.nansum(liwcPosValues)
            # sdLiwcPos = numpy.nanstd(liwcPosValues)
        else:
            sumLIWCPos = ''
            # sdLiwcPos = ''
        if len(liwcNegValues) > 0:
            sumLIWCNeg = numpy.nansum(liwcNegValues)
            # sdLiwcNeg = numpy.nanstd(liwcNegValues)
        else:
            sumLIWCNeg = ''
            # sdLiwcNeg = ''
        if len(warrinerValenceValues) > 0:
            meanValence_w = numpy.nanmean(warrinerValenceValues)
            # sdValence_w = numpy.nanstd(warrinerValenceValues)
        else:
            meanValence_w = ''
            # sdValence_w = ''
        if len(warrinerArousalValues) > 0:
            meanArousal_w = numpy.nanmean(warrinerArousalValues)
            # sdArousal_w = numpy.nanstd(warrinerArousalValues)
        else:
            meanArousal_w = ''
            # sdArousal_w = ''
        if len(anewValenceValues) > 0:
            meanValence_a = numpy.nanmean(anewValenceValues)
            # sdValence_a = numpy.nanstd(anewValenceValues)
        else:
            meanValence_a = ''
            # sdValence_a = ''
        if len(anewArousalValues) > 0:
            meanArousal_a = numpy.nanmean(anewArousalValues)
            # sdArousal_a = numpy.nanstd(anewArousalValues)
        else:
            meanArousal_a = ''
            # sdArousal_a = ''

        # allFeatureValues = [wordCount, meanSentiment_b, sdSentiment_b,
        #                     sumLIWCPos, sdLiwcPos, sumLIWCNeg, sdLiwcNeg,
        #                     meanValence_w, sdValence_w, meanArousal_w, sdArousal_w,
        #                     meanValence_a, sdValence_a, meanArousal_a, sdArousal_a]
        allFeatureValues = [wordCount, meanSentiment_b, #sdSentiment_b,
                            sumLIWCPos, #sdLiwcPos, 
                            sumLIWCNeg, #sdLiwcNeg,
                            meanValence_w, #sdValence_w, 
                            meanArousal_w, #sdArousal_w,
                            meanValence_a, #sdValence_a, 
                            meanArousal_a]#, sdArousal_a]

        if self.useGlove:
            wordVec = self.extractWordVectorFromWords(words)
            allFeatureValues.extend(wordVec)

        return allFeatureValues



def decodeAscii(text):
    # replace unicode characters with plain ascii
    # after http://stackoverflow.com/questions/15321138/removing-unicode-u2026-like-characters-in-a-string-in-python2-7
    replacements = {u'\u2019': "\'",
                    u'\u2026': "...",
                    u'\u201c': "\"",
                    u'\u201d': "\""}
    textDecoded = text.decode('utf-8')
    for key in replacements:
        textDecoded = textDecoded.replace(key, replacements[key])
    return unidecode(textDecoded)  #.encode('ascii')
    #return text.decode('utf-8').replace(u'\u2019', "\'").replace(u'\u2026', "...").replace(u'\u201c', "\"").replace(u'\u201d', "\"").encode('ascii')


def replaceUnknownWords(text):
    # Unfortunately, the forced alignment tool I'm using will just crap out
    # if there's an unknown word
    # so this function is a hacky way to fix a list of unknown words
    # by replacing them with:
    #     (i) a similar, unconjugated version, e.g., "shitty" -> "shit"
    #     (ii) homophones (this is mostly for proper nouns like NuvaRing, Gringotts...)
    # .replace("\"", "").replace("\'", "")

    # replacements = {"memory's": "memory is",
    #                 "NuvaRing": "noon ring",
    #                 "swingy": "swing",
    #                 "shitty": "shit",
    #                 "construing": "construe",
    #                 "gra..": "grasp",
    #                 "don\'t\'": "don\'t",
    #                 "Kunming": "could min",
    #                 "Yunnan": "you none",
    #                 "Jamba": "jump a",
    #                 "insanely": "insane lee",
    #                 "Hogwarts": "hog warts",
    #                 "Diagon": "die gone",
    #                 "hogsmeade": "Hog mead",
    #                 "Gringotts": "green gods",
    #                 "wizarding": "wizard thing",
    #                 "goas": "go as",
    #                 "episiotomy": "epic to me",
    #                 "out-of-body": "out of body",
    #                 "scariness": "scary ness",
    #                 "moveable": "move able",
    #                 "reallysupport": "really support",
    #                 "Ender\'s": "End the",
    #                 "crystalized": "crystal ice",
    #                 "Mauian": "Maui Ian",
    #                 "Napili": "Na pill",
    #                 "Fountainhead": "Fountain head",
    #                 "Tonga": "Tong a",
    #                 "SAT\'s": "SAT",
    #                 "hobbit": "hob it",
    #                 "layed": "laid",
    #                 "fiftied": "fifty",
    #                 "anxs": "ants",
    #                 "titling": "title",
    #                 "Popsicles": "popsicle",
    #                 "aunties": "aunt",
    #                 "auntie": "aunt",
    #                 "overdosing": "overdose",
    #                 " IU": " I U",  # indiana university??
    #                 "besties": "best",
    #                 "carpooled": "carpool",
    #                 "VP": "V P",  # vice president?
    #                 "empathizing": "empathize",
    #                 "empathi...": "empathy...",  # stutter
    #                 "Mountain Camp 2": "Mountain Camp Two",  # changing numeral
    #                 " classmen": " class men", # from upper classmen
    #                 " positivevery": " positive very",  # missing space
    #                 " 5:00 am": " five am",
    #                 " 2009" : " twenty oh nine",
    #                 "babysitter's": "babysitter",
    #                 " 911": " nine one one",
    #                 " premonitions": " prey monique",
    #                 " ACL": " A C L",
    #                 "trainer's": "trainers",
    #                 "pivoting": "pivot",
    #                 "rehabbed": "rehab",
    #                 "mindfulness": "mindful ness",
    #                 "trajectories": "trajectory",
    #                 "far\'": "far",
    #                 "traumatizing": "traumatic",
    #                 "RoHo": "Row Ho",
    #                 "dysregulated": "this regulated",
    #                 "Cuddles": "cuddle",
    #                 "hav...": "have...",  # stutter
    #                 "heav,": "heavy,",  # stutter
    #                 "humors": "humor",
    #                 "aloneness": "alone ness",
    #                 "rageful": "rage full",
    #                 "elementaries": "elementary",
    #                 "animalistic": "animal list",
    #                 "Belle's": "Belle",
    #                 "unitards": "uni tough",
    #                 "MIT's": "MIT is",
    #                 "admission's": "admissions",
    #                 "exaltation": "exalt station",
    #                 "grandparent's": "grandparents",
    #                 "snowmobiling": "snowmobile",
    #                 " 2004": " two thousand four",
    #                 "crappy": "crap",
    #                 " LDS": " L D S",
    #                 "2008": "two thousand eight",
    #                 "ICU": "I C U",
    #                 "ridiculousness": "ridiculous ness",
    #                 "4th": "fourth",
    #                 " 2014": " twenty fourteen",
    #                 "blog": "log",
    #                 "Netflix": "net flicks",
    #                 " UC ": " U C ",
    #                 "thousand-dollar": "thousand dollar",
    #                 "29": "twenty nine",
    #                 " 2015": " twenty fifteen",
    #                 " 2010": " twenty ten", # 126_vid3
    #                 " AP ": " A P ",
    #                 "env..." : "and",  # stutter
    #                 " 8" : " eight",
    #                 "2012": "twenty twelve",
    #                 "7:00": "seven",  # am
    #                 "brang": "brought",  # weird word choice?
    #                 "Pally...": "Pale...",  # stutter
    #                 "whippings": "whipping",
    #                 "uh...\"Middle": "uh... \"Middle",
    #                 "loneliest": "loneliness",  # closest homonym?
    #                 "kickin\'": "kicking",
    #                 "startin\'": "starting",
    #                 "Whattup": "whats up",
    #                 "crackheads": "crack heads",
    #                 "brainstem": "brain stem",
    #                 "\'cause...": "\'cause... ",
    #                 " HR": " H R",
    #                 "groomsmen": "grooms men",
    #                 "Y\'know": "you know",
    #                 "y\'know": "you know",
    #                 "racecars": "race cars",
    #                 "sorta": "sort of",
    #                 "ulti-": "alt ",  # stutter
    #                 "Lucille's": "Lucy",  # proper noun
    #                 "Zeke's": "leek",  # proper noun
    #                 " TCE": " T C E",
    #                 "Pokemon": "po key mon",
    #                 "carefreeness": "carefree ness",
    #                 "puttin\'": "putting",
    #                 "2016": "twenty sixteen",
    #                 "jotting": "jogging",
    #                 "5th": "fifth",
    #                 "foolin\'": "fooling",
    #                 "seein\'": "seeing",
    #                 "suicide/depression": "suicide depression",
    #                 "offin\'": "off ing",
    #                 "tiffs": "tiff",
    #                 "shithead": "shit head",
    #                 "school/middle": "school middle",
    #                 "depression\'s": "depression is",
    #                 " od\'ed": " O D D",
    #                 "Bourdain": "Boar Dame",
    #                 "Frairi": "Fry eerie",
    #                 "Fieri": "Fiery",
    #                 " mmm": " umm",
    #                 "zookeeper" : "zoo keeper",
    #                 "longboarding" : "long board ing",
    #                 "freakin\'": "freaking",
    #                 "chillin\'": "chilling",
    #                 " A/B" : " A B",
    #                 "Um...\'cause": "um... cause",
    #                 "Usain": "U say",
    #                 "Uggs": "ark",  #...
    #                 " meh": " heh",
    #                 "growin\'": "growing",
    #                 "routine": "root ing",
    #                 "Blackketter": "black cat her", # proper name?
    #                 "macros": "macro",
    #                 "priorit...": "priority",  # stutter
    #                 "opportun...": "opportune",  # stutter
    #                 "2000\'s": "two thousands",
    #                 "don\' ": "don\'t ", # typo?
    #                 "laundrymat": "laundry mat",
    #                 "crockpot": "crock pot",
    #                 "hoarder": "hoard er",
    #                 "Tiana": "T Anna",
    #                 "YouTuber": "YouTube er",
    #                 " RA\'s": " R A is",
    #                 " H2 " : " H two ",
    #                 " G2 " : " G two ",
    #                 "procrastinator": "procrastinate",
    #                 " 2011": " twenty eleven",
    #                 " 11:59": " eleven fifty nine",  # pm
    #                 "January 1": "January First",
    #                 " UCLA": " U C L A",
    #                 " January 25": " January twenty five",
    #                 " 2006": " Two thousand six",
    #                 "Bodleian": "Bore the lane",
    #                 "sixty-five-year-old": "sixty five year old",
    #                 "Sappy\'s": "Sap E",
    #                 "touristy": "tourist",
    #                 "Mins...": "Mint...",  # stutter
    #                 "artsy-fartsy": "artsy far see",
    #                 "housewarming": "house warming",
    #                 " \"Bruh": " \"bro",
    #                 " Ooo": " woo",
    #                 "hundred-dollar": "hundred dollar",
    #                 " UK": " U K",
    #                 " Hungar...": " hunger...", # stutter for Hungary
    #                 "Braaa!": "bra",
    #                 " sh-she": " she she",
    #                 " f --": " ", # transcriber's way of writing stuttering onsets
    #                 " t --": " ", # transcriber's way of writing stuttering onsets
    #                 " y --": " ", # transcriber's way of writing stuttering onsets
    #                 " s --,": " ", # transcriber's way of writing stuttering onsets
    #                 " c -- --": " ", # transcriber's way of writing stuttering onsets
    #                 "jus --": "just ", # transcriber's way of writing stuttering onsets
    #                 "w --...": "",  # stutter
    #                 "Sc, uh,": "sir", # stutter
    #                 " --, ": " ",
    #                 " --'": " ",
    #                 "o-on": "on on",
    #                 "m-my": "my my",
    #                 "a-a": "a a",
    #                 "s'mores": "more",
    #                 "K-12": "K twelve",
    #                 "freshman's": "freshman",
    #                 "bas, basically": "base, basically ", # stutter
    #                 "wedding's": "wedding is",
    #                 "bedsheets": "bed sheets",
    #                 "takeaway": "take away",
    #                 "actualization": "actualize",
    #                 "transitioned": "transition",
    #                 "defeatus": "defeatist",
    #                 " TAs": " T A",
    #                 " CS106A": " C S one oh six A",
    #                 " 106B": " one oh six B",
    #                 " 106X": " one oh six X",
    #                 "undergrads": "undergrad",
    #                 "recursion": "re curse",
    #                 " route": " root",
    #                 " CS": " C S",
    #                 "anime": "animate",
    #                 "snorkeling": "snore curling",
    #                 "fa...": "far", # stutter
    #                 "floaty": "float",
    #                 "hypoallergenic": "hypo allergen",
    #                 "whisperer": "whisper",
    #                 "grandpa\'s": "grandpa",
    #                 "boyfriend\'s": "boyfriend",
    #                 "telekinesis": "tele kinetic",
    #                 "messaged": "message",
    #                 "Prius": "pre us",
    #                 "Yunque": "Yoon Key",  # El Yunque the rainforest
    #                 " Mm.": " Um.",  # not too good...
    #                 "tonsillitis": "tonsil light this",
    #                 " texted": " text",
    #                 " crossly": " cross",
    #                 " impactful": " impact full",
    #                 " Ivies": " ivy",
    #                 " segway\'s": " segue",
    #                 " inbox": " in box",
    #                 " munchkin": " munch kin",
    #                 " Fetty Wap": " Fatty Wa",  # last name?
    #                 " PSAT": " P S A T",
    #                 " Gabby\'s": " Gabby",
    #                 " commencements": " commencement",
    #                 " GPA": " G P A",
    #                 " Yorkie": " york key",
    #                 " chocolaty": " chocolate",
    #                 " bawling": " balling",
    #                 " Mm,": " Um,",
    #                 " funner": " funnel",
    #                 " fessing": " fest ing",
    #                 " HBO": " H B O",
    #                 " NFL": " N F L",
    #                 " workout-aholic": " workout hall lick"
    #                 }

    replacements = {
        "2:11" : "two eleven", # ID120_vid2_transcribe.txt->2:11 
        "2:09" : "two o nine", # ID120_vid2_transcribe.txt->2:09 
        " 2015" : " twenty fifteen", # ID127_vid6_transcribe.txt->2015 
        " 2010" : " twenty ten", # ID130_vid1_transcribe.txt->2010 
        " 2011": " twenty eleven", # ID130_vid2_transcribe.txt->2011 
        " 2013": " twenty thirteen", # ID180_vid3_transcribe.txt->2013
        " --": " ",
        " --, ": " ",
        " --. ": " ",
        " --! ": " ",
        " --? ": " ",
        " --\" ": " ",
        " --\' ": " ",
    }
    for key in replacements:
        text = text.replace(key, replacements[key])
    replacements = {
        "2:11" : "two eleven", # ID120_vid2_transcribe.txt->2:11 
        "2:09" : "two o nine", # ID120_vid2_transcribe.txt->2:09 
        " 2015" : " twenty fifteen", # ID127_vid6_transcribe.txt->2015 
        " 2010" : " twenty ten", # ID130_vid1_transcribe.txt->2010 
        " 2011": " twenty eleven", # ID130_vid2_transcribe.txt->2011 
        " 2013": " twenty thirteen", # ID180_vid3_transcribe.txt->2013
        " 29": " twenty nine",
        " 7:00": " seven",
        " 911": " nine eleven",
        " 2016": " twenty sixteen",
        " 5th": " fifth",
        " 2000's": " two thousands",
        " 25": " twenty five",
        " 2006" : " two thousands six",
        " CS106A": "CS one o six A",
        " 106B": " one o six B",
        " 106X": " one o six X",
    }
    for key in replacements:
        text = text.replace(key, replacements[key])
    return text


# from text_to_transcript import text_to_transcript
# import text_to_transcript
def text_to_transcript(text_file, output_file):
    '''
    This function is originally from https://github.com/ucbvislab/p2fa-vislab
    I had trouble importing this function and getting it to work,
    so I just copied the whole function here
    '''
    text = open(text_file).read()

    filedir = os.path.dirname(os.path.realpath(__file__))
    schema_path = os.path.join(filedir, "helper/p2faVislab/alignment-schemas/transcript_schema.json")

    transcript_schema = json.load(open(schema_path))

    paragraphs = text.split("\n\n")
    out = []
    for para in paragraphs:
        para = para.replace("\n", " ")

        # added this line to replace unicode chars
        # para = decodeAscii(para)

        if para == "" or para.startswith("#"):
            continue

        print para

        line = {
            "speaker": "Target",
            "line": para
        }
        out.append(line)

    jsonschema.validate(out, transcript_schema)
    if output_file is None:
        print json.dumps(out, indent=4)
    else:
        with open(output_file, 'w') as f:
            f.write(json.dumps(out, indent=4))
    return


def calculateFeaturesFromJSON(jsonFilename, outputFilename, myFeatureExtractor, windowSize=5):
    '''
    usage:
    e.g.
    calculateFeaturesFromJSON('videoProcessing/testOutput/902_vid1_forcedAligned.json',
        '902/ID902_vid1_words_within_window.tsv')
    '''
    json_data = json.loads(open(jsonFilename).read())

    #myFeatureExtractor = linguisticFeatureExtractor()

    currentStartTime = 0
    featureNames = myFeatureExtractor.features
    tsv = 'time\twords\t' + '\t'.join(featureNames) + '\n'
    thisString = ''
    for thisWord in json_data['words']:
        if thisWord['start'] > currentStartTime:
            if thisWord['word'][0]!='{':
                thisString += thisWord['word'] + " "  # only include non special chars: {}
        if thisWord['end'] > (currentStartTime + windowSize):
            textFeatures = myFeatureExtractor.calculateFeaturesFromText(thisString)
            tsv += str(currentStartTime) + "\t" + thisString + "\t" + '\t'.join([str(x) for x in textFeatures]) + "\n"
            currentStartTime += windowSize
            thisString = ''
    parsed = open(outputFilename, 'w')
    parsed.write(tsv)
    parsed.close()



def removeTimestampsFromTranscripts(transcriptInput, transcriptOutput):
    # relevant only for transcripts that have a [00:00:01] timecodes
    # at least for now, I want to just remove them to have a plaintext file
    inputFile = open(transcriptInput).read()
    # removing the [00:00:00] timecodes
    scrubbedText = re.sub('\[[0-5][0-9]\:[0-5][0-9]\:[0-5][0-9]\] ', '', inputFile)
    scrubbedText = re.sub('\[.*?\]', '', scrubbedText)
    # forcing a space after periods if there isn't.
    scrubbedText = re.sub("\.([a-zA-Z])", ". \\1", scrubbedText)
    # removing spaces before .
    scrubbedText = re.sub("\s\.", ".", scrubbedText)
    # forcing a space before or after a "-"
    scrubbedText = re.sub("\-\-([a-zA-Z])", "-- \\1", scrubbedText)
    scrubbedText = re.sub("([a-zA-Z])\-\-", "\\1 --", scrubbedText)
    # removing floating punctuations: but just ,.!?
    scrubbedText = re.sub(' [.,?!]+', '', scrubbedText) # space, then punctuation
    scrubbedText = re.sub('^[.,?!]+', '', scrubbedText) # at start of string
    # cleaning up unicode and unknown words
    # scrubbedText = replaceUnknownWords(decodeAscii(scrubbedText))
    scrubbedText = replaceUnknownWords(scrubbedText)
    scrubbedFile = open(transcriptOutput, 'w')
    print "Success! Scrubbed timestamps from " + transcriptInput + " and wrote output to " + transcriptOutput + "."
    scrubbedFile.write(scrubbedText)
    scrubbedFile.close()


def convertGloveToPickledDictionary(numDimensions=50):
    # takes in the trained word vector representation
    # (GloVe 6Billion tokens corpus: https://nlp.stanford.edu/projects/glove/)
    # converts it to a lookup dictionary
    # and pickles the dictionary
    if numDimensions not in [50, 100, 200, 300]:
        print "Error, num dimensions not recognized (must be 50, 100, 200, or 300)"
        return
    inputFilename = "models/glove.6B/glove.6B." + str(numDimensions) + "d.txt"
    pickledOutputName = PICKLED_MODEL_DIRECTORY + "glove" + str(numDimensions) + ".p"
    vectorDict = {}
    with open(inputFilename, "r") as fileDescriptor:
        lines = fileDescriptor.read().split('\n')
    for line in lines:
        values = line.split(' ')
        vectorDict[values[0]] = [numpy.float(j) for j in values[1:]]

    pickle.dump(vectorDict, open(pickledOutputName, "wb"))



def examineGloveDimensions():
    # takes in a factor loading of glove weights
    # prints out the words that have the highest dot product with that factor
    # from R code: 
    # "glove40"   1.517058e-03
    # "glove44"   -1.643024e-03
    # "glove69"   -6.154625e-03      
    # "glove80"   -9.307186e-04
    # "glove121"   2.010386e-04
    # "glove130"   -6.420586e-03
    # "glove156"   -1.725925e-05
    # "glove180"   -7.687437e-04
    # "glove221"   5.794650e-03  
    # "glove226"   8.670613e-03  
    # "glove249"   2.744338e-04
    weightedFactor = numpy.zeros(300)
    weightedFactor[40] = 1.517058e-03
    weightedFactor[44] = -1.643024e-03
    weightedFactor[69] = -6.154625e-03
    weightedFactor[80] = -9.307186e-04
    weightedFactor[121] = 2.010386e-04
    weightedFactor[130] = -6.420586e-03
    weightedFactor[156] = -1.725925e-05
    weightedFactor[180] = -7.687437e-04
    weightedFactor[221] = 5.794650e-03 
    weightedFactor[226] = 8.670613e-03 
    weightedFactor[249] = 2.744338e-04
    glove = pickle.load(open(PICKLED_MODEL_DIRECTORY + "glove300.p", "r"))
    scores = {}
    for word in glove:
        scores[word] = numpy.sum(glove[word] * weightedFactor)

    # from http://pythoncentral.io/how-to-sort-python-dictionaries-by-key-or-value/
    sorted(scores, key=scores.__getitem__, reverse=True)














