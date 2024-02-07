import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import spacy
# nltk.download('punkt')
# nltk.download('stopwords')

# Load the spacy model
nlp = spacy.load("en_core_web_sm")
# Load the stop words
stop_words = set(stopwords.words("english"))
# Load the stemmer
stemmer = SnowballStemmer("english")

# Function to perform word tokenization
def wordTokenize(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return tokens

# Function to perform line tokenization
def lineTokenize(text):
    tokens = text.split('\n')
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return list(set(tokens))

# Function to perform space tokenization
def spaceTokenize(text):
    tokens = text.split()
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return tokens

# Function to perform tweet tokenization
def tweetTokenize(text):
    from nltk.tokenize import TweetTokenizer
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    newTokens = []
    for word in tokens:
        if(word.isalpha()):
            newTokens.append(word.lower())
        elif ord(word) > 100000:
            newTokens.append(word)
        else:
            print(f"{word} : {ord(word)}")

    return newTokens


# Function to perform stemming
def stem(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens
# Function to perform lemmatization
def lemmatize(tokens):
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    return lemmatized_tokens
# Function to remove stop words
def remove_stop_words(tokens):
    filtered_tokens = [word for word in tokens if not word in stop_words]
    return filtered_tokens

def calculateTTR(tokens):
    total = len(tokens)
    unique = len(set(tokens))
    return unique/total

# Creating corpus from Text File
def createCorpusFromText():
    # Load the text from the doc file
    with open("document.txt") as file:
        text = file.read()
    return text

# Creating corpus from Docx File
def createCorpusFromDocx():
    import docx
    # load the document
    doc = docx.Document('document.docx')
    # extract the content
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

# Creating corpus from PDF File
def createCorpusFromPdf():
    import PyPDF2
    # open the PDF file in read binary mode
    pdf_file = open('document.pdf', 'rb')
    # create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    # get the number of pages
    num_pages = len(pdf_reader.pages)
    #pdf_reader.numPages

    # initialize an empty string to store the text
    text = ''
    # loop through each page and extract the text
    for i in range(num_pages):
        page = pdf_reader.pages[i]
        #pdf_reader.getPage(i)
        text += page.extract_text()
        #page.extractText()
    return text

# Creating corpus from Website
def createCorpusFromWebsite():
    import requests
    from bs4 import BeautifulSoup
    # get the website content
    url = 'https://www.google.com'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    return text

print("Select one option from below to create a Corpus \n 1.Text \n 2.Docx \n 3.PDF \n 4.Website")
option = int(input())
if(option == 1):
    text = createCorpusFromText()
elif(option == 2):
    text = createCorpusFromDocx()
elif(option == 3):
    text = createCorpusFromPdf()
else:
    text = createCorpusFromWebsite()
    

# Tokenize the text
tokens = word_tokenize(text)
# Perform stemming
stemmed_tokens = stem(tokens)
# Perform lemmatization
lemmatized_tokens = lemmatize(tokens)
# Remove stop words
filtered_tokens = remove_stop_words(lemmatized_tokens)
# Print the tokens
print("Original Tokens:", tokens[:20])
print("Stemmed Tokens:", stemmed_tokens[:20])
print("Lemmatized Tokens:", lemmatized_tokens[:20])
print("Filtered Tokens:", filtered_tokens[:20])
print("TTR:", calculateTTR(filtered_tokens))