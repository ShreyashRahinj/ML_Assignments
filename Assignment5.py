from nltk import RegexpParser,sent_tokenize,word_tokenize,pos_tag

text = "The boy holds tightly."

grammar = '\n'.join([
    # 'NP : {<DT>*<NNP>}',
    # 'NP : {<JJ>*<NN>}',
    # 'NP : {<NNP>+}',
    # 'NP : {<DT>?<JJ>*<NN>}',
    # 'ADJP : {<JJ>*<NN>}',
    # 'ADJP :{<JJ>*<CC>*<JJ>}',
    'VP: {<VB.?><DT>?<JJ>*<NN><RB.?>?}',
    'VP: {<DT>?<JJ>*<NN><VB.?><RB.?>?}',
])

sentences = sent_tokenize(text)

for sentence in sentences:
    words = word_tokenize(sentence)
    tags = pos_tag(words)
    chunkparser = RegexpParser(grammar)
    result = chunkparser.parse(tags)
    result.draw()