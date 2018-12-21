import linguisticProcessing
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import os

def main():
	# list input docx files
    input_dir_transcript="./transcript_txt/"
    inpput_dir_audio_wav="./audio_wav/"
    input_transcript_dir_list=[]
    output_transcript_dir_list=[]
    input_audio_wav_dir_list=[]
    txt_files = [f for f in listdir(input_dir_transcript) if isfile(join(input_dir_transcript, f))]
    for file in txt_files:
    	if file[0:3] == ".DS" or not file[-3:] == 'txt':
    		continue
        prefix = file.split(".")[0]
    	input_transcript_dir_list.append(input_dir_transcript + file)
    	output_transcript_dir_list.append(input_dir_transcript + prefix.split("_")[0] + "_" + prefix.split("_")[1] + ".txt")
    	input_audio_wav_dir_list.append(inpput_dir_audio_wav + prefix.split("_")[0] + "_" + prefix.split("_")[1] + ".wav")
    # print len(input_transcript_dir_list)
    # print input_audio_wav_dir_list

    # cleanup_file = [f for f in listdir(input_dir_transcript) if isfile(join(input_dir_transcript, f))]
    # print cleanup_file

    # for count in range(0, 90):
    # 	# for count in range(0, len(input_transcript_dir_list)):
    # 	linguisticProcessing.removeTimestampsFromTranscripts(input_transcript_dir_list[count], output_transcript_dir_list[count])
    # 	linguisticProcessing.process(output_transcript_dir_list[count], input_audio_wav_dir_list[count])

    cleanup_file = [f for f in listdir(input_dir_transcript) if isfile(join(input_dir_transcript, f))]
    # # clean-up, only get the aligned json
        
    for file in cleanup_file:
    	if file.split("_")[-1] == "aligned.json":
    		aligned_dir="./transcript_txt/aligned_txt/"
    		if not os.path.exists(aligned_dir):
    			os.makedirs(aligned_dir)
    		copyfile(join(input_dir_transcript, file), join(aligned_dir, file))

    # for file in cleanup_file:
    # 	if not file.split("_")[-1] == "transcribe.txt":
    # 		try:
    # 			os.remove(join(input_dir_transcript, file))
    # 		except OSError:
    #   			pass

if __name__ == "__main__":
    main()