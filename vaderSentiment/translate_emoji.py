from google.cloud import translate_v2 as translate
translate_client = translate.Client()


source = open("emoji_utf8_lexicon.txt", 'r')
output = open("emoji.txt", 'w')

for line in source.readlines():
    #print("line[0] ->" + line.split('\t')[0] + "\nline[1] ->" + line.split('\t')[1].strip() + "\n")
    #output_line = line.split('\t')[0] + "\t" + line.split('\t')[1].strip() + "\n"
    #print(output_line)
    result = translate_client.translate(
        line.split('\t')[1].strip(), target_language='pl')
    #print(result)
    if result['detectedSourceLanguage'] == 'en':
        output_line = line.split('\t')[0].strip() + "\t" + result['translatedText'].strip() + "\n"
    else:
        output_line = line
    print(line + output_line)
    output.write(output_line)

