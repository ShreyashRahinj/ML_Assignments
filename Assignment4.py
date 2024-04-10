import nltk
from nltk import pos_tag, word_tokenize, TnT
from nltk.corpus import treebank, indian
nltk.download('indian')


def learnRETagger(simpleSentence):
    customPatterns = [
        (r'.*ing$', 'ADJECTIVE'),
        (r'.ly$', 'ADVERB'),
        (r'.ion$', 'NOUN'),
        (r'(.*ate|.*en|is)$', 'VERB'),
        (r'^an$', 'INDEFINITE-ARTICLE'),
        (r'^(with|on|at)$', 'PREPOSITION'),
        (r'^\-?[0-9]+(\.[0-9]+)$', 'NUMBER'),
        (r'.$', None)
    ]

    tagger = nltk.RegexpTagger(customPatterns)
    wordsInSentence = nltk.word_tokenize(simpleSentence)
    posEnabledTags = tagger.tag(wordsInSentence)
    print(posEnabledTags)


def learnLookupTagger(simpleSentence):
    mapping = {
        '.': '.', 'place': 'NN', 'on': 'IN', 'earth': 'NN', 'Mysore': 'NNP', 'is': 'VBZ', 'an': 'DT', 'amazing': 'JJ'
    }

    tagger = nltk.UnigramTagger(model=mapping)
    wordsInSentence = nltk.word_tokenize(simpleSentence)
    posEnabledTags = tagger.tag(wordsInSentence)
    print(posEnabledTags)


def tagger(sentence):
    train_sentences = treebank.tagged_sents()[:2500]
    bigram_tagger = nltk.UnigramTagger(train_sentences)
    tagged_words = bigram_tagger.tag(word_tokenize(sentence))
    print(tagged_words)


def tagger_marathi(sentence):
    train_data = indian.tagged_sents()
    tnt_pos_tagger = TnT()
    tnt_pos_tagger.train(train_data)
    tagged_words = tnt_pos_tagger.tag(nltk.word_tokenize(sentence))
    print(tagged_words)


# Inbuilt Tagger
sentence = "Mysore is an amazing place on Earth. I have visited Mysore 10 times"
print(pos_tag(word_tokenize(sentence)))

# Regex Tagger
learnRETagger(sentence)

# Lookup Tagger
learnLookupTagger(sentence)

# Unigram Tagger
tagger(sentence)

# Marathi
text1 = "मी पाणी पिते."
text2 = "मी पुस्तक वाचत होते."
tagger_marathi(text2)