import string
import requests
import codecs
import json

def get_base(data):
    response = requests.post('http://localhost:9200/?output_format=conll', data=data.encode('utf-8'))

    if response.status_code != 200:
        print("requesting server failed")

    string_response = str(response.content, 'utf8')
    words = string_response.split("\\")
    words_splitted = words[0].split()
    # print(words_splitted)
    list_of_skipped_words = ["disamb", "none", "space", "conj", "interp", "newline", "comp", "qub", "adv", "pred"]
    for word in words_splitted:
        if word in list_of_skipped_words or ":" in word:
            words_splitted.remove(word)

    # second iteration, because the first one doesn't remove "dismab" words
    for word in words_splitted:
        if word in list_of_skipped_words or ":" in word:
            words_splitted.remove(word)

    # getting every second element as those are tagged words
    words_splitted = words_splitted[1::2]
    words_splitted = ' '.join(map(str, words_splitted))

    return words_splitted


source = open("polish_vader_dictionary.txt", 'r')
output = open("base_polish.txt", 'w')

for line in source.readlines():
    result = get_base(line.split('\t')[0])
    output_line = result + "\t" + '\t'.join(line.split('\t')[1:])

    print(line + output_line)
    output.write(output_line)




