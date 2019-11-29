# coding: utf-8
# Author: C.J. Hutto
# Thanks to George Berry for reducing the time complexity from something like O(N^4) to O(N).
# Thanks to Ewan Klein and Pierpaolo Pantone for bringing VADER into NLTK. Those modifications were awesome.
# For license information, see LICENSE.TXT

"""
If you use the VADER sentiment analysis tools, please cite:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""
import os
import re
import math
import string
import requests
import codecs
import json
from itertools import product
from inspect import getsourcefile
from io import open
import requests

# ##Constants##

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

# (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
C_INCR = 0.733
N_SCALAR = -0.74

NEGATE = \
    ["nie", "ani", "nic", "pomimo", "mimo", "rzadko", "niekiedy", "sporadycznie",
     "nigdzie", "bez", "nigdy", "żaden", "żadny", "żadne", "żadna", "niczego", "nikt",
     "niczego"]
# booster/dampener 'intensifiers' or 'degree adverbs'
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs
""" "effing": B_INCR, "frackin": B_INCR, "fracking": B_INCR,  "frigging": B_INCR, "friggin": B_INCR, 
     "fuckin": B_INCR, "fucking": B_INCR, "fuggin": B_INCR, "fugging": B_INCR,"""

BOOSTER_DICT = \
    {"absolutnie": B_INCR, "niezwykle": B_INCR, "okropnie": B_INCR,
     "zupełnie": B_INCR, "całkowicie": B_INCR, "całkiem": B_INCR, "znacznie": B_INCR, "znacząco": B_INCR,
     "dedydująco": B_INCR, "głęboko": B_INCR,  "ogromnie": B_INCR, "olbrzymie": B_INCR,
     "kompletnie": B_INCR, "szczególnie": B_INCR, "wyjątkowy": B_INCR, "wyjątkowo": B_INCR,
     "skrajny": B_INCR, "skrajnie": B_INCR, "zwłaszcza": B_INCR,
     "wspaniale": B_INCR, "cholernie": B_INCR, "cholerny": B_INCR,
     "zajebiście": B_INCR, "zajebisty": B_INCR, "w pełni": B_INCR,
     "zdecydowanie": B_INCR, "wysoce": B_INCR, "wielce": B_INCR,
     "niezwykły": B_INCR, "niezwykle": B_INCR, "intensywnie": B_INCR, "nieziemsko": B_INCR,
     "główny": B_INCR, "głównie": B_INCR, "więcej": B_INCR, "bardziej": B_INCR, "w większości": B_INCR, "w szczególności": B_INCR,
     "jedynie": B_INCR, "dosyć": B_INCR, "serio": B_INCR, "znacząco": B_INCR,
     "tak": B_INCR, "istotnie": B_INCR, "zasadniczo": B_INCR,
     "dogłębnie": B_INCR, "zupełny": B_INCR, "totalnie": B_INCR, "oblrzymi": B_INCR, "niesamowicie": B_INCR,
     "niesamowity": B_INCR, "ogromny": B_INCR, "mega": B_INCR, "ultra": B_INCR,
     "super": B_INCR, "niewyobrażalnie": B_INCR, "niezwyczajnie": B_INCR, "absolutny": B_INCR, "bezgraniczny": B_INCR,
     "bardzo": B_INCR, "bezgranicznie": B_INCR,
     "prawie": B_DECR, "ledwo": B_DECR, "ledwie": B_DECR, "z trudem": B_DECR, "zaledwie": B_DECR, "prawie wcale": B_DECR,
     "z trudnością": B_DECR, "z ledwością": B_DECR, "nieco": B_DECR, "dość": B_DECR,
     "jakby": B_DECR, "tak jakby": B_DECR, "poniekąd": B_DECR, "niby": B_DECR, "tylko": B_DECR,
     "mniej": B_DECR, "trochę": B_DECR, "marginalny": B_DECR, "marginalnie": B_DECR, "niewielki": B_DECR,
     "niewiele": B_DECR, "niezbyt": B_DECR, "nieznacznie": B_DECR, "nieznaczny": B_DECR,
     "sporadyczny": B_DECR, "sporadycznie": B_DECR, "częściowo": B_DECR, "niewielki": B_DECR, "troszeczkę": B_DECR,
     "niewystarczająco": B_DECR, "niedostatecznie": B_DECR, "trochę": B_DECR, "troszkę": B_DECR, "w pewnym stopniu": B_DECR,
     "raczej": B_DECR, "w pewnym sensie": B_DECR, "trochę jakby": B_DECR}

# check for sentiment laden idioms that do not contain lexicon words (future work, not yet implemented)
SENTIMENT_LADEN_IDIOMS = {"cut the mustard": 2, "hand to mouth": -2,
                          "back handed": -2, "blow smoke": -2, "blowing smoke": -2,
                          "upper hand": 1, "break a leg": 2,
                          "cooking with gas": 2, "in the black": 2, "in the red": -2,
                          "on the ball": 2, "under the weather": -2}

# check for special case idioms containing lexicon words
SPECIAL_CASE_IDIOMS = {"the shit": 3, "the bomb": 3, "bad ass": 1.5, "badass": 1.5,
                       "yeah right": -2, "kiss of death": -1.5, "to die for": 3}


# #Static methods# #

def negated(input_words, include_nt=True):
    """
    Determine if input contains negation words
    """
    input_words = [str(w).lower() for w in input_words]
    neg_words = []
    neg_words.extend(NEGATE)
    for word in neg_words:
        if word in input_words:
            return True
    if include_nt:
        for word in input_words:
            if "nie" in word:
                return True
    '''if "least" in input_words:
        i = input_words.index("least")
        if i > 0 and input_words[i - 1] != "at":
            return True'''
    return False


def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score


def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different


def scalar_inc_dec(word, valence, is_cap_diff):
    """
    Check if the preceding words increase, decrease, or negate/nullify the
    valence
    """
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT:
        scalar = BOOSTER_DICT[word_lower]
        if valence < 0:
            scalar *= -1
        # check if booster/dampener word is in ALLCAPS (while others aren't)
        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar


def simplify_polish_words(sentence):
    response = requests.post('http://localhost:9200/?output_format=conll', data=sentence.encode('utf-8'))
    if response.status_code != 200:
        print("requesting server failed")

    string_response = str(response.content, 'utf8')
    words = string_response.split("\\")
    words_splitted = words[0].split()

    list_of_skipped_words = ["disamb", "none", "space", "conj", "interp", "newline", "comp", "qub", "adv",
                                 "pred"]
    for word in words_splitted:
        if word in list_of_skipped_words or ":" in word:
            words_splitted.remove(word)

        # second iteration, because the first one doesn't remove "disamb" w
    for word in words_splitted:
        if word in list_of_skipped_words or ":" in word:
            words_splitted.remove(word)

        # getting every second element as those are tagged words
    words_splitted = words_splitted[1::2]
    simplified_sentence = ' '.join(map(str, words_splitted))

    return simplified_sentence


class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        self.base_words = self._base_words()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    @staticmethod
    def _strip_punc_if_word(token):
        """
        Removes all trailing and leading punctuation
        If the resulting string has two or fewer characters,
        then it was likely an emoticon, so return original string
        (ie ":)" stripped would be "", so just return ":)"
        """
        stripped = token.strip(string.punctuation)
        if len(stripped) <= 2:
            return token
        return stripped

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        stripped = list(map(self._strip_punc_if_word, wes))
        return stripped

    def _base_words(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = simplify_polish_words(self.text).split()
        stripped = list(map(self._strip_punc_if_word, wes))
        return stripped


class SentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="polish_vader_dictionary.txt", emoji_lexicon="emoji.txt", base_file="base_polish.txt"):
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), lexicon_file)
        with codecs.open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        base_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), base_file)
        with codecs.open(base_full_filepath, encoding='utf-8') as f:
            self.base_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()
        self.base = self.make_base_dict()


        emoji_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), emoji_lexicon)
        with codecs.open(emoji_full_filepath, encoding='utf-8') as f:
            self.emoji_full_filepath = f.read()
        self.emojis = self.make_emoji_dict()

    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.rstrip('\n').split('\n'):
            if not line:
                continue
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict

    def make_base_dict(self):
        """
        Convert base file to a dictionary
        """
        base_dict = {}
        for line in self.base_full_filepath.rstrip('\n').split('\n'):
            if not line:
                continue
            (word, measure) = line.strip().split('\t')[0:2]
            base_dict[word] = float(measure)
        return base_dict

    def make_emoji_dict(self):
        """
        Convert emoji lexicon file to a dictionary
        """
        emoji_dict = {}
        for line in self.emoji_full_filepath.rstrip('\n').split('\n'):
            (emoji, description) = line.strip().split('\t')[0:2]
            emoji_dict[emoji] = description
        return emoji_dict

    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # convert emojis to their textual descriptions
        text_no_emoji = ""
        prev_space = True
        for chr in text:
            if chr in self.emojis:
                # get the textual description
                description = self.emojis[chr]
                if not prev_space:
                    text_no_emoji += ' '
                text_no_emoji += description
                prev_space = False
            else:
                text_no_emoji += chr
                prev_space = chr == ' '
        text = text_no_emoji.strip()

        sentitext = SentiText(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for i, item in enumerate(words_and_emoticons):
            valence = 0
            # check for vader_lexicon words that may be used as modifiers or negations
            if item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue
            #kind of -> tak jakby
            if (i < len(words_and_emoticons) - 1 and item.lower() == "tak" and
                    words_and_emoticons[i + 1].lower() == "jakby"):
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        sentiments = self._but_check(words_and_emoticons, sentiments)

        valence_dict = self.score_valence(sentiments, text)

        return valence_dict

    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        base_words = sentitext.base_words
        item_lowercase = item.lower()
        if item_lowercase in self.lexicon:
            # get the sentiment valence
            valence = self.lexicon[item_lowercase]

            # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
            if item_lowercase == "nie" and \
                    (words_and_emoticons[i + 1].lower() in self.lexicon or base_words[i + 1].lower() in self.base):
                # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0
            if (i > 0 and words_and_emoticons[i - 1].lower() == "nie") \
                    or (i > 1 and words_and_emoticons[i - 2].lower() == "nie") \
                    or (i > 2 and words_and_emoticons[i - 3].lower() == "nie" and words_and_emoticons[i - 1].lower() in [
                "lub", "ani"]):
                valence = self.lexicon[item_lowercase] * N_SCALAR

            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower()\
                        not in self.lexicon and base_words[i - (start_i + 1)].lower() not in self.base:
                    s = scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._special_idioms_check(valence, words_and_emoticons, i)

            if i > 0 and item.lower() == "przynajmniej":
                valence = valence * N_SCALAR

        elif base_words[i].lower() in self.base:
            item_lowercase = base_words[i].lower()
            # get the sentiment valence
            valence = self.base[item_lowercase]

            # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
            if item_lowercase == "nie" and\
                    (words_and_emoticons[i + 1].lower() in self.lexicon or base_words[i + 1].lower() in self.base):
                # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0
            if (i > 0 and base_words[i - 1].lower() == "nie") \
                    or (i > 1 and base_words[i - 2].lower() == "nie") \
                    or (
                    i > 2 and base_words[i - 3].lower() == "nie" and base_words[i - 1].lower() in [
                "lub", "ani"]):
                valence = self.base[item_lowercase] * N_SCALAR

            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower()\
                        not in self.lexicon and base_words[i - (start_i + 1)].lower() not in self.base:
                    s = scalar_inc_dec(base_words[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._special_idioms_check(valence, words_and_emoticons, i)

            if i > 0 and item.lower() == "przynajmniej":
                valence = valence * N_SCALAR
        sentiments.append(valence)
        return sentiments

    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        # check for modification in sentiment due to contrastive conjunction 'but'
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if 'ale' in words_and_emoticons_lower:
            bi = words_and_emoticons_lower.index('ale')
            for sentiment in sentiments:
                si = sentiments.index(sentiment)
                if si < bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 0.5)
                elif si > bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 1.5)
        return sentiments

    @staticmethod
    def _special_idioms_check(valence, words_and_emoticons, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        onezero = "{0} {1}".format(words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoonezero = "{0} {1} {2}".format(words_and_emoticons_lower[i - 2],
                                          words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoone = "{0} {1}".format(words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwoone = "{0} {1} {2}".format(words_and_emoticons_lower[i - 3],
                                           words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwo = "{0} {1}".format(words_and_emoticons_lower[i - 3], words_and_emoticons_lower[i - 2])

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]

        for seq in sequences:
            if seq in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[seq]
                break

        if len(words_and_emoticons_lower) - 1 > i:
            zeroone = "{0} {1}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1])
            if zeroone in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[zeroone]
        if len(words_and_emoticons_lower) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1],
                                              words_and_emoticons_lower[i + 2])
            if zeroonetwo in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[zeroonetwo]

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        n_grams = [threetwoone, threetwo, twoone]
        for n_gram in n_grams:
            if n_gram in BOOSTER_DICT:
                valence = valence + BOOSTER_DICT[n_gram]
        return valence

    @staticmethod
    def _sentiment_laden_idioms_check(valence, senti_text_lower):
        # Future Work
        # check for sentiment laden idioms that don't contain a lexicon word
        idioms_valences = []
        for idiom in SENTIMENT_LADEN_IDIOMS:
            if idiom in senti_text_lower:
                print(idiom, senti_text_lower)
                valence = SENTIMENT_LADEN_IDIOMS[idiom]
                idioms_valences.append(valence)
        if len(idioms_valences) > 0:
            valence = sum(idioms_valences) / float(len(idioms_valences))
        return valence


#TODO to be adjusted to polish
    @staticmethod
    def _negation_check(valence, words_and_emoticons, start_i, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if start_i == 0:
            if negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 1 word preceding lexicon word (w/o stopwords)
                valence = valence * N_SCALAR
        if start_i == 1:
            if words_and_emoticons_lower[i - 2] == "nigdy" and \
                    (words_and_emoticons_lower[i - 1] == "więc" or
                     words_and_emoticons_lower[i - 1] == "dlatego" or
                     words_and_emoticons_lower[i - 1] == "to"):  # dunno if translated correctly
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 2] == "bez" and \
                    (words_and_emoticons_lower[i - 1] == "wątpienia" or
                     words_and_emoticons_lower[i - 1] == "wątpliwości"):
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 2 words preceding the lexicon word position
                valence = valence * N_SCALAR
        if start_i == 2:
            if words_and_emoticons_lower[i - 3] == "nigdy" and \
                    words_and_emoticons_lower[i - 2] == "nie":

                valence = valence * 1.25
                #print(words_and_emoticons[i] + str(valence))
            elif words_and_emoticons_lower[i - 3] == "bez" and \
                    ((words_and_emoticons_lower[i - 2] == "wątpienia" or
                      words_and_emoticons_lower[i - 2] == "wątpliwości")
                     or (words_and_emoticons_lower[i - 1] == "wątpienia" or
                        words_and_emoticons_lower[i - 1] == "wątpliwości")):
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 3 words preceding the lexicon word position
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    @staticmethod
    def _amplify_ep(text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    @staticmethod
    def _amplify_qm(text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    @staticmethod
    def _sift_sentiment_scores(sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        #print(sentiments)
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg": round(neg, 3),
             "neu": round(neu, 3),
             "pos": round(pos, 3),
             "compound": round(compound, 4)}

        return sentiment_dict


# if __name__ == '__main__':
#     # --- examples -------
#     sentences = ["VADER jest mądrym, przystojna i zabawny.",  # positive sentence example
#                  "VADER jest mądry, przystojny i zabawny!!",
#                  # punctuation emphasis handled correctly (sentiment intensity adjusted)
#                  "VADER jest bardzo mądry, przystojny i zabawny!",
#                  # booster words handled correctly (sentiment intensity adjusted)
#                  "VADER jest BARDZO MĄDRY, przystojny i zabawny!",  # emphasis for ALLCAPS handled
#                  "VADER jest BARDZO MĄDRY, przystojny i ZABAWNY!!!!!!",
#                  # combination of signals - VADER appropriately adjusts intensity
#                  "VADER jest BARDZO MĄDRY, ultra przystojny i  NIEZIEMSKO ZABAWNY!!!!!!",
#                  # booster words & punctuation make this close to ceiling for score
#                  "VADER nie jest mądry, przystojny, ani zabawny.",  # negation sentence example
#                  "Książka była dobra.",  # positive sentence
#                  "Przynajmniej nie jest okropny.",  # negated negative sentence with contraction
#                  "Książka była tylko trochę dobra.",
#                  "To był bez wątpienia najlepszy obiad w moim życiu",
#                  "To był najlepszy obiad w moim życiu",
#                  # qualified positive sentence is handled correctly (intensity adjusted)
#                  "Fabuła była spoko, ale postacie słabe i zdjęcią okropne",
#                  "Ale on jest chujem",
#                  # mixed negation sentence
#                  "Dzisiaj jest nieco niemiły dzień!",  # negative slang with capitalization emphasis
#                  "Dzisiaj jest niemiły dzień",
#                  "Tamta kobieta mnie  strasznie denerwuje",
#                  "Tamta kobieta mnie nieco denerwuje",
#                  # mixed sentiment example with slang and constrastive conjunction "but"
#                  "Pamiętaj żeby :) albo :D dzisiaj!",  # emoticons handled
#                  "Catch utf-8 emoji such as 💘 and 💋 and 😁",  # emojis handled
#                  "Całkiem nie najgorzej",
#                  """My, Polacy, mamy najlepszy i najgodniejszy Prezydent wszechczasów – to Andrzej Duda !
#                  Tyle co ten człowiek tylko przez kilka lat zrobił dla Polski – to ogromne osiągnięcia.
#                  Nie będę wymieniał poszczególnych jego zasługa, bo lista jest bardzo długa, ale w dziedzinach takich jak
#                   bezpieczeństwo Polski i pokój na świecie, służba Polakom, reprezentowanie państwa polskiego,
#                    propagowanie przedsiębiorczości i biznesu, obronności kraju oraz rozwój Polski i
#                   Europy nasz Prezydent brał czynny, intensywny i systematyczny udział ! """# Capitalized negation
#                  ]
#
#
#     analyzer = SentimentIntensityAnalyzer()
#
#
#     for sentence in sentences:
#         #sentence = simplify_polish_words(sentence)
#         vs = analyzer.polarity_scores(sentence)
#         print("{:-<65} {}".format(sentence, str(vs)))
#     print("----------------------------------------------------")
#     print(" - About the scoring: ")
#     print("""  -- The 'compound' score is computed by summing the valence scores of each word in the lexicon, adjusted
#      according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
#      This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence.
#      Calling it a 'normalized, weighted composite score' is accurate.""")
#     print("""  -- The 'pos', 'neu', and 'neg' scores are ratios for proportions of text that fall in each category (so these
#      should all add up to be 1... or close to it with float operation).  These are the most useful metrics if
#      you want multidimensional measures of sentiment for a given sentence.""")
#     print("----------------------------------------------------")
#
#     input("\nPress Enter to continue the demo...\n")  # for DEMO purposes...
#
#     tricky_sentences = ["Analiza sensdtymentu nigdy nie była dobra.",
#                         "Analiza sentsdymentu nigdy nie była TAK dobra!",
#                         "Większość automatycznych narzędzi analizy sentymentu jest do dupy",
#                         "Z VADEREM, analiza sentymentu to jest to!",
#                         "Inne narzędzia analizy sentymentu są dość słabe",
#                         "Z drugiej strony, Vader jest dość dobry",
#                         "Vader jest fajowy!!",  # slang with punctuation emphasis
#                         "Bez wątpienia, świetny pomysł",
#                         "Jednak nie taki dobry",  # Capitalized negation with slang
#                         ]
#     print("----------------------------------------------------")
#     print(" - Analyze examples of tricky sentences that cause trouble to other sentiment analysis tools.")
#     print("  -- special case idioms - e.g., 'never good' vs 'never this good', or 'bad' vs 'bad ass'.")
#     print("  -- special uses of 'least' as negation versus comparison \n")
#     for sentence in tricky_sentences:
#         vs = analyzer.polarity_scores(sentence)
#         print("{:-<69} {}".format(sentence, str(vs)))
#     print("----------------------------------------------------")
#
#     print("\n\n Demo Done!")