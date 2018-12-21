import sys
import os
from os import listdir
from os.path import isfile, join
import re
import shutil

input_dir = "./transcript_txt"
output_dir = "./processed_transcript_txt"
txt_files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
for file in txt_files:
	if file[-1] == 't':
		with open(join(input_dir, file), 'r') as myfile:
			data = myfile.read()
			data = re.sub(r'[^\x00-\x7F]+','', data)
			with open(join(output_dir, file), 'w') as f:
				f.write(data)
