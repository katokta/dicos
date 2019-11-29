import dicos
from dicos import fetch_ocr, fetch_ocr_vectorized
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tkinter import *
from gooey import Gooey, GooeyParser

#loading training dataset
twenty_train = fetch_ocr(subset='train', categories=None, shuffle=True, random_state=42)

#defining classification method
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
count_vect.vocabulary_.get(u'algorithm')

#shaping the model
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

#fitting the algorithm to the model
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

def getResults(filename):
     """
     Parameters:

     filename = PATH of the file to be classified

     ============================================
     Returns : a dictionary filled with machine learning decision

     """
     dic={}
     docs=open(filename, 'r')
     docs=docs.readline()
     docs=docs.split('\n')
     docs_new_counts=count_vect.transform(docs)
     docs_new_tfidf=tfidf_transformer.transform(docs_new_counts)
     predict=clf.predict(docs_new_tfidf)
     for doc, category in zip(docs, predict):
          dic[doc]=twenty_train.target_names[category]
     return dic

def saveResults(dics): #dics is dictionary
     """
     Parameters:

     dics = dictionary filled with raw decision of the classification
     =============================================
     Returns : string containing the final decision
     """
     val={}
     results=dics.values()
     for result in results:
          if result in val:
               val[result]+=1
          else:
               val[result]=1
     values=val.values()
     most=max(val.values())
     words=[]
     for k in val:
          if val[k]==most:
               words.append(k)
     label=''
     for item in words:
          label+=item
     return label


@Gooey(program_name = "Pilih File untuk digolongkan")
def main():
     """
     main loop to call the appropriate function to the GUI for input
     """
     ap=GooeyParser()
     ap.add_argument("-fn","--filename", metavar="File", help="Pilih nama file yang mau digolongkan (.txt)", widget="FileChooser")
     args=ap.parse_args()
     filename=args.filename
     dics=getResults(filename)
     reading=saveResults(dics)
     file=os.path.basename(filename)
     print("%s adalah %s" % (file,reading))
     
main()
