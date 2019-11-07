import string
import requests
import codecs
import json
from google.cloud import translate_v2 as translate
translate_client = translate.Client()


source = open("vader_lexicon.txt", 'r')
output = open("test3.txt", 'w')

for line in source.readlines():

    result = translate_client.translate(
        line.split('\t')[0], target_language='pl')
    #print(result)
    if result['detectedSourceLanguage'] == 'en':
        output_line = result['translatedText'] + "\t" + line.split('\t')[1] + "\t" + line.split('\t')[2] + "\t" + line.split('\t')[3]
    else:
        output_line = line
    print(line + output_line)
    output.write(output_line)

