import string
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import string


def stopwords_removal(n, a):
    b = []
    if n == 1:
        for word in a:
            count = 0
            if word in stop_words:
                count = 0
            else:
                count = 1
            if count == 1:
                b.append(word)
        return b
    else:
        for pair in a:
            count = 0
            for word in pair:
                if word in stop_words:
                    count = count or 0
                else:
                    count = count or 1
            if count == 1:
                b.append(pair)
        return b


def get_ngrams_freqDist(n, ngramList):
    ngram_freq_dict = {}
    for ngram in ngramList:
        if ngram in ngram_freq_dict:
            ngram_freq_dict[ngram] += 1
        else:
            ngram_freq_dict[ngram] = 1
    return ngram_freq_dict


def predict_next_word(last_word, probDist):
    next_word = {}
    for k in probDist:
        if k[0] == last_word[0]:
            next_word[k[1]] = probDist[k]
    k = Counter(next_word)
    high = k.most_common(1)
    return high[0]


def predict_next_3_words(token, probDist):
    pred1 = []
    pred2 = []
    next_word = {}
    for i in probDist:
        if i[0] == token:
            next_word[i[1]] = probDist[i]
    k = Counter(next_word)
    high = k.most_common(2)
    w1a = high[0]
    w1b = high[1]
    w2a = predict_next_word(w1a, probDist)
    w3a = predict_next_word(w2a, probDist)
    w2b = predict_next_word(w1b, probDist)
    w3b = predict_next_word(w2b, probDist)
    pred1.append(w1a)
    pred1.append(w2a)
    pred1.append(w3a)
    pred2.append(w1b)
    pred2.append(w2b)
    pred2.append(w3b)
    return pred1, pred2


string.punctuation = string.punctuation + '“' + '”' + '-' + '’' + '‘' + '—'
string.punctuation = string.punctuation.replace('.', '')
file = open('sherlock.txt').read()

# preprocess data to remove newlines and special characters
file_new = ""
for line in file:
    line_new = line.replace("\n", " ")
    file_new += line_new
preprocessedCorpus = "".join([char for char in file_new if char not in string.punctuation])

sentences = sent_tokenize(preprocessedCorpus)
print("1st 5 sentences of preprocessed corpus are : ")
print(sentences[0:5])
words = word_tokenize(preprocessedCorpus)
print("1st 5 words/tokens of preprocessed corpus are : ")
print(words[0:5])

stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in words if not w.lower() in stop_words]

unigrams = []
bigrams = []
trigrams = []
for content in sentences:
    content = content.lower()
    content = word_tokenize(content)
    for word in content:
        if word == '.':
            content.remove(word)
        else:
            unigrams.append(word)
    bigrams.extend(ngrams(content, 2))
    trigrams.extend(ngrams(content, 3))
print("Sample of n-grams:n" + "-------------------------")
print("--> UNIGRAMS: n" + str(unigrams[:5]) + " ...n")
print("--> BIGRAMS: n" + str(bigrams[:5]) + " ...n")
print("--> TRIGRAMS: n" + str(trigrams[:5]) + " ...n")

unigrams_Processed = stopwords_removal(1, unigrams)
bigrams_Processed = stopwords_removal(2, bigrams)
trigrams_Processed = stopwords_removal(3, trigrams)
print("Sample of n-grams after processing:n" + "-------------------------")
print("--> UNIGRAMS: n" + str(unigrams_Processed[:5]) + " ...n")
print("--> BIGRAMS: n" + str(bigrams_Processed[:5]) + " ...n")
print("--> TRIGRAMS: n" + str(trigrams_Processed[:5]) + " ...n")

unigrams_freqDist = get_ngrams_freqDist(1, unigrams)
unigrams_Processed_freqDist = get_ngrams_freqDist(1, unigrams_Processed)
bigrams_freqDist = get_ngrams_freqDist(2, bigrams)
bigrams_Processed_freqDist = get_ngrams_freqDist(2, bigrams_Processed)
trigrams_freqDist = get_ngrams_freqDist(3, trigrams)
trigrams_Processed_freqDist = get_ngrams_freqDist(3, trigrams_Processed)

smoothed_bigrams_probDist = {}
V = len(unigrams_freqDist)
for i in bigrams_freqDist:
    smoothed_bigrams_probDist[i] = (bigrams_freqDist[i] + 1) / (unigrams_freqDist[i[0]] + V)
smoothed_trigrams_probDist = {}
for i in trigrams_freqDist:
    smoothed_trigrams_probDist[i] = (trigrams_freqDist[i] + 1) / (bigrams_freqDist[i[0:2]] + V)

testSent1 = "There was a sudden jerk, a terrific convulsion of the limbs; and there he"
testSent2 = "They made room for the stranger, but he sat down"
testSent3 = "The hungry and destitute situation of the infant orphan was duly reported by"

token_1 = word_tokenize(testSent1)
token_2 = word_tokenize(testSent2)
token_3 = word_tokenize(testSent3)
ngram_1 = {1: [], 2: []}
ngram_2 = {1: [], 2: []}
ngram_3 = {1: [], 2: []}
for i in range(2):
    ngram_1[i + 1] = list(ngrams(token_1, i + 1))[-1]
    ngram_2[i + 1] = list(ngrams(token_2, i + 1))[-1]
    ngram_3[i + 1] = list(ngrams(token_3, i + 1))[-1]
print("Sentence 1: ", ngram_1, "nSentence 2: ", ngram_2, "nSentence 3: ", ngram_3)

print("Predicting next 3 possible word sequences with smoothed bigram model : ")
pred1, pred2 = predict_next_3_words(ngram_1[1][0], smoothed_bigrams_probDist)
print("1a)" + testSent1 + " " + pred1[0][0] + " " + pred1[1][0] + " " + pred1[2][0])
print("1b)" + testSent1 + " " + pred2[0][0] + " " + pred2[1][0] + " " + pred2[2][0])
pred1, pred2 = predict_next_3_words(ngram_2[1][0], smoothed_bigrams_probDist)
print("2a)" + testSent2 + " " + pred1[0][0] + " " + pred1[1][0] + " " + pred1[2][0])
print("2b)" + testSent2 + " " + pred2[0][0] + " " + pred2[1][0] + " " + pred2[2][0])
pred1, pred2 = predict_next_3_words(ngram_3[1][0], smoothed_bigrams_probDist)
print("3a)" + testSent3 + " " + pred1[0][0] + " " + pred1[1][0] + " " + pred1[2][0])
print("3b)" + testSent3 + " " + pred2[0][0] + " " + pred2[1][0] + " " + pred2[2][0])
