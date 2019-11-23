import string
import requests
import codecs
import json
from google.cloud import translate_v2 as translate
translate_client = translate.Client()


source = open("polish_vader_dictionary.txt", 'r')
output = open("with_space.txt", 'w')

for line in source.readlines():

    first = line.split('\t')[0]
    words = first.split(' ')
    if len(words) > 1:
        output.write(line)
    # if result['detectedSourceLanguage'] == 'en':
    #     output_line = result['translatedText'] + "\t" + line.split('\t')[1] + "\t" + line.split('\t')[2] + "\t" + line.split('\t')[3]
    # else:
    #     output_line = line
    # print(line + output_line)
    # output.write(output_line)

