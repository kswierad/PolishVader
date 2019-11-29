from polish_vader_sentiment import SentimentIntensityAnalyzer



loop = True
while loop:
    analyzer = SentimentIntensityAnalyzer()
    sentence = input("Input sentence you want to analyze:\n")
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))
    end = input("Type 'y' if you want to continue.\n")
    loop = end == "y"
