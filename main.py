from nltk.corpus import stopwords 
from nltk.corpus.reader import reviews
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tag import CRFTagger
from pprint import pprint

reviews = [
    "apa nama pesantren yang dibangun oleh sunan ampel ?",
    "siapakah anggota wali songo ?",
    "kapan sunan gresik lahir ?",
    "dimana daerah asal sunan gresik ?",
    "tahun berapa islam masuk ke nusantara ?"
]

stop_words = set(stopwords.words("indonesian"))

#tokenizing
word_tokens = []
for review in reviews:
    word_tokens.append(word_tokenize(review))

#case folding
casefolded_sentence = []
for word_token in word_tokens:
    casefolded_sentence.append([word.casefold() for word in word_token])

#stop word removal
filtered_sentence = []
for sent in casefolded_sentence:
    filtered_sentence.append([word for word in sent if not word in stop_words])

#parse list of word
sentences = []
for filtered_sent in filtered_sentence:
    sentences.append(' '.join(filtered_sent))

#stemming
stemmer = StemmerFactory().create_stemmer()
stemmed_sentence = []
for sentence in sentences:
    stemmed_sentence.append(stemmer.stem(sentence).split(" "))

#pos tagging
ct = CRFTagger()
ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')
results = ct.tag_sents(stemmed_sentence)
pprint(results)