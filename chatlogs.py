import glob
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
import nltk
from collections import Counter
import gensim
from gensim import corpora
from gensim.models import LdaModel

# The following script extracts the most important topics from the given .tsv file.
# please run this script in a python notebook. 

# The comments with '# def' are for further fine-tuning the code to run as apython program.


#getting the data from a .tsv file and writing it ot a separate file for processing.
#def get_data(path):
path = '/Users/salil/Downloads/dialogs/4/*.tsv'     
files=glob.glob(path)
with open('/Users/salil/Downloads/dialogs/chat.txt', 'w') as outfile:
    for fname in files:
        with open(fname) as infile:
            outfile.write(infile.read())
    
# Converting the file into a dataframe and extratcing the text column to perform pre processing tasks like tokenization,
# stripping blank spaces,lower casing of strings.Stopwords were used to clean the text.

def preprocess(df):
df=pd.read_csv('/Users/salil/Downloads/dialogs/chat.txt', sep='\t')
df.columns = ["TimeStamp", "From", "To", "chatlogs"]
df['filtered_sentences'] = df['chatlogs'].str.replace(r'[!|$|@|#|%|^|*|<|>|/|\|"|=|+|_|~|`|:|,|()|\d+|{|}|\n|\t|?|,|.|[|]|\'|]','')
sentences = df['filtered_sentences'].str.replace('-','')
sentences = sentences.astype('str')
sentences_lower = [x.lower() for x in sentences]
sentences = [a.strip() for a in sentences_lower]
    
words_token = [ word_tokenize(s) for s in sentences ]
    
stopwords_en = stopwords.words('english')
stopwords_eng = [ a.replace('\'','') for a in stopwords_en]
stopwords_english = [ str(w) for w in stopwords_eng ]

word_final = []
word_final_str = []

for i in words_token:
    word_final.append(list(set(i)-set(stopwords_english)))

for s in word_final:
    for w in s:
        word_final_str.append(w)

            
# Using the above pre-processed text, we stem and lemmatize the words.

#def stem_lemmatize(word_final_str): 
word_encode = [ unicode(w, errors='replace') for w in word_final_str ]
porter = PorterStemmer()
word_stem = [ porter.stem(word) for word in word_encode ]
lem = WordNetLemmatizer()
word_lem = [ lem.lemmatize(word) for word in word_stem ]

relevant_words = word_lem
# Using the LDA approach for topic modelling. Converting to a dictionary of frequently used words by 
# creating a sparse matrix of words inorder of importance.

#def create_dct_lda(relevant_words):
dictionary = corpora.Dictionary([relevant_words])
dictionary.save('dictionary.dict')
doc_term_matrix = [dictionary.doc2bow(doc) for doc in [relevant_words]]
corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)

#def create_lda_model(doc_term_matrix):
Lda = gensim.models.ldamodel.LdaModel
    # Solution for Question 1
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)
for i in ldamodel.print_topics(): 
    for j in i: print j
ldamodel.save('topic.model')
            
#def find_relevant_topics():
model = LdaModel.load('topic.model')

# for a new .tsv file please follow the above steps till we create a doc_term_matrix and pass it to the
# lda model created above. 

# relevance_model = model[test] 
# predict = pd.DataFrame(test,columns=['id','prob']).sort_values('prob',ascending=False)
# predict['topic'] = predict['id'].apply(model.print_topic)
# predict ['topic']
