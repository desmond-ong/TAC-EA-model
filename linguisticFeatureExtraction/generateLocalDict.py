import sys
import os
from os import listdir
from os.path import isfile, join
import re
import shutil
# from g2p_en import g2p
# All words containing digits: ID117_vid3.txt->2 ID127_vid3.txt->29 ID127_vid4.txt->7:00 ID129_vid7.txt->911 ID131_vid2.txt->2016 ID131_vid2.txt->2016 ID131_vid3.txt->5TH ID137_vid1.txt->2000'S ID137_vid5.txt->25 ID137_vid5.txt->2006 ID153_vid3.txt->2006 ID156_vid6.txt->CS106A ID156_vid6.txt->106B ID156_vid6.txt->106X
def replaceUnknownWords(text):
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
    scrubbedText = replaceUnknownWords(scrubbedText)
    scrubbedFile = open(transcriptOutput, 'w')
    print ("Success! Scrubbed timestamps from " + transcriptInput + " and wrote output to " + transcriptOutput + ".")
    scrubbedFile.write(scrubbedText)
    scrubbedFile.close()

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def cleanUp(dir):
	shutil.rmtree(dir)

def deSegmentDict(dictInput, segDictOutput):
    with open(dictInput) as _dict:
    	words = [line.strip('\n').split("  ")[0] for line in _dict]
    char_li = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(' ')
    char_li_ind = 0
    tmp_words = []
    for word in words:
    	if char_li_ind <= 25 and word[0] == char_li[char_li_ind]:
    		if not word[0] == "A":
    			if not os.path.exists(segDictOutput):
    				os.makedirs(segDictOutput)
    			with open(join(segDictOutput, "dict_" + char_li[char_li_ind - 1]), "w") as f:
    				for tmp_word in tmp_words:
    					f.write(tmp_word + '\n')
    		tmp_words = []
    		char_li_ind = char_li_ind + 1
    	tmp_words.append(word)
    # append Z
    with open(join(segDictOutput, "dict_Z"), "w") as f:
    	for tmp_word in tmp_words:
    		f.write(tmp_word + '\n')
    print ("Success! Segment the dict from " + dictInput + " and wrote output to " + segDictOutput + ".")

def main():
	# print command line arguments
    arg_li = [arg for arg in sys.argv[1:]]
    if not len(arg_li) == 2:
    	print ("USAGE: python generateLocalDict.py <INPUT_DIR> <DICT_DIR>")
    	return
    input_dir = arg_li[0]
    dict_dir = arg_li[1]
    output_dir = "./dict/dict.local"
    # list input docx files
    txt_files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    # parsed file temp dict
    tmp_dir = "./_tmp/"
    if not os.path.exists(tmp_dir):
    	os.makedirs(tmp_dir)
    words_list_lookup_dir = join(tmp_dir, 'words2lookup.txt')
    words_list_output_dir = join(tmp_dir, 'words_g2p.txt')
    g2p_model_dir = "./g2p_model"

   	# de-segment dict for easier read in memory
    deSegmentDict(dict_dir, tmp_dir + "_dict/")

    # avoid repeat g2p lookup
    word_memo=[]
    # to look at digits
    digits_word_memo=[]
    digits_file_memo=[]

    # srubbed the origin transcript to remove timestampes
    for file in txt_files:
    	if not file[-3:] == 'txt':
    		continue
    	print ("Performing G2P for unknown words in transcript: " + file)
    	scrubbed_txt_file = file.split('_')[0] + '_' + file.split('_')[1] + "_scrubbed.txt"
    	removeTimestampsFromTranscripts(join(input_dir, file), join(tmp_dir, scrubbed_txt_file))
    	txt_line = []
    	with open(join(tmp_dir, scrubbed_txt_file)) as f:
    		txt_line = f.readlines()[0].strip(" ")
        # TODO: i think these steps have to match what we have in align.py
        txt_line = re.sub(r"([A-Za-z])\.\.\.(.*)", r"\1... \2", txt_line)

    	words_orig = txt_line.split(" ")
    	words_dict_lookup = [word.strip('?:!.,;\"\'-').upper() for word in txt_line.split(" ")] # <- used for dict checking
    	words_g2p_lookup = [re.sub(r'[^\w]', '', word).upper() for word in txt_line.split(" ")] # <- used for g2p
    	# print (words_dict_lookup)
    	# print (len(words_g2p_lookup))

    	# here we need to add special case for
    	# this pattern matches hyphenated words, such as TWENTY-TWO; however, it doesn't work with longer things like SOMETHING-OR-OTHER
    	hyphenPat = re.compile(r'([a-zA-Z]+)-([a-zA-Z]+)')
    	hyphenWords=[]
    	for word in words_dict_lookup:
    		if hyphenPat.match(word):
    			new_wrd = re.sub(hyphenPat, r'\1 \2', word)
    			new_wrd = new_wrd.split()
    			hyphenWords.extend(new_wrd)

        # we add more corner case cover for ... case
        

        	# break up any hyphenated words into two separate words
            # new_wrd = re.sub(hyphenPat, r'\1 \2', word)
            # new_wrd = new_wrd.split()
            # if len(new_wrd) > 1:
            # 	print (new_wrd)
            # hyphenWords.extend(new_wrd)

    	words_dict_lookup.extend(hyphenWords)
    	# check if the word exist in the dict
    	for i in range(0, len(words_dict_lookup)):
    		if hasNumbers(words_dict_lookup[i].strip(" ")):
    			digits_file_memo.append(file)
    			digits_word_memo.append(words_dict_lookup[i].strip(" "))
    		if not len(words_dict_lookup[i].strip(" ")) == 0 and not hasNumbers(words_dict_lookup[i].strip(" ")):
    			# print (words_dict_lookup[i])
    			word2lookup = words_dict_lookup[i].strip(" ")
    			dict_name = "dict_" + word2lookup[0]
    			with open(join(tmp_dir + "_dict/", dict_name)) as _dict:
    				dict_words = _dict.readlines()
    			dict_words = [word.strip() for word in dict_words]
    			if not word2lookup in dict_words and not word2lookup in word_memo:
    				print ("G2P Transcript: " + file + ". Word: " + word2lookup + ".")
    				word_memo.append(word2lookup)
    				if os.path.exists(words_list_lookup_dir):
    				    append_write = 'a' # append if already exists
    				else:
    				    append_write = 'w' # make a new file if not
    				with open(words_list_lookup_dir, append_write) as words_f:
    					words_f.write(word2lookup.lower()+"\n")

    # generate our own dict for un-known words to cmudict using g2p open-source
    command = "g2p-seq2seq --decode "+ words_list_lookup_dir +" --model_dir "+ g2p_model_dir + " --output " + words_list_output_dir
    print (command)
    os.system(command)

    with open(words_list_output_dir) as f_r:
    	words_g2p_list = f_r.readlines()
    words_g2p_list = [x.strip() for x in words_g2p_list]

    with open(output_dir, 'w') as f_w:
	    for tu in words_g2p_list:
	    	f_w.write(tu.split(" ")[0].upper() + "  " + " ".join(tu.split(" ")[1:]) + "\n")
	    	print("G2P Lookup Word: " + tu.split(" ")[0].upper() + ". G2P Results: " + " ".join(tu.split(" ")[1:]))

	# print all digits word
    digits_pair_print=[]
    for i in range(0, len(digits_word_memo)):
    	digits_pair_print.append(digits_file_memo[i] + "->" + digits_word_memo[i])
    print ("All words containing digits: " + " ".join(digits_pair_print))

    print ("Success! G2P lookup finished. Local dict is generated in " + output_dir + ".")

    try:
    	os.remove(words_list_lookup_dir)
    	os.remove(words_list_output_dir)
    except OSError:
    	pass

   	cleanUp(tmp_dir)

if __name__ == "__main__":
    main()