import streamlit as st
import pandas as pd
import numpy as np
from numpy import array
import pickle
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import GaussianNB

Data, Implementasi = st.tabs(['Introduction & Data', 'Implementasi'])

def preproses(inputan):
         clean_tag = re.sub('@\S+','', inputan)
         clean_url = re.sub('https?:\/\/.*[\r\n]*','', clean_tag)
         clean_hastag = re.sub('#\S+',' ', clean_url)
         clean_symbol = re.sub('[^a-zA-Z]',' ', clean_hastag)
         casefolding = clean_symbol.lower()
         token=word_tokenize(casefolding)
         listStopword = set(stopwords.words('indonesian')+stopwords.words('english'))
         stopword=[]
         for x in (token):
            if x not in listStopword:
               stopword.append(x)
         joinkata = ' '.join(stopword)
         return clean_symbol,casefolding,token,stopword,joinkata


with Data :
   st.title("""Pencarian & Penambangan Web A""")
   st.text('Nama : Abd. Hanif Azhari')
   st.text('NIM  :  200411100101')
   st.subheader('Deskripsi Data')
   st.write("""Fitur pada dataset yang dicrawling dari detik.com dan pojokbaca.id :""")
   st.text("""
            1) Judul
            2) Isi
            3) Kategori""")
   st.subheader('Data Gabungan')
   data=pd.read_csv('crawling_berita_gabungan.csv')
   data
   
   st.write("""Dari data hasil crawling yang sudah dilakukan sebelumnya dilakukan beberapa tahapan pemrosesan
             sebelum sampai pada tahap perhitungan akurasi dan implementasi terdiri dari :""")
   st.write("""
            **1) Preprocessing**    : 
           Dalam tahapan preprocessing terdiri dari beberapa tahapan yang diantaranya cleansing, casefolding/ lowercase, tokenize dan stopword removal.
           
            **2) Ekstraksi Fitur**  : 
           Pada tahapan ekstraksi fitur melakukan proses perhitungan frekuensi kata pada suatu dokumen atau yang disebut term frequency.
            
            **3) Splitting Data**   : 
           Pada tahapan splitting data ini dilakukan proses pembagian data penelitian menjadi 2 bagian yaitu data training dan testing dengan perbandingan data 80:20.
            
            **4) Latent Dirichlet Allocation (LDA)**    : 
           Pada tahapan LDA ini melakukan proses pelatihan data training dari hasil splitting dengan uji coba pada jumlah topik 1 hingga 50 topik untuk mendapatkan akurasi terbaik.
            
            **5) Modelling**    : 
           Tahapan modelling merupakan proses untuk mencari nilai akurasi dengan menerapkan hasil dari uji coba LDA pada topik 1 hingga 50.""")

with Implementasi:
    st.title("""Implementasi Data""")
    inputan = st.text_input('Masukkan Isi Berita')
    submit = st.button("Submit")
    if submit :
        clean_symbol,casefolding,token,stopword,joinkata = preproses(inputan)

        with open('tf.sav', 'rb') as file:
            vectorizer = pickle.load(file)
        
        hasil_tf = vectorizer.transform([joinkata])
        tf_name=vectorizer.get_feature_names_out()
        tf_array=hasil_tf.toarray()
        df_tf= pd.DataFrame(tf_array, columns = tf_name)

        with open('lda.sav', 'rb') as file:
            lda = pickle.load(file)

        hasil_lda=lda.transform(df_tf)   

        with open('naive.sav', 'rb') as file:
            naive = pickle.load(file)
        
        hasil_naive=naive.predict(hasil_lda)
        hasil =f"Berdasarkan data yang Anda masukkan, maka berita masuk dalam kategori  : {hasil_naive}"
        st.success(hasil)

        st.subheader('Preprocessing')
        st.markdown('**Cleansing :**')
        clean_symbol
        st.markdown('**Casefolding :**')
        casefolding
        st.markdown('**Tokenisasi :**')
        token
        st.markdown("**Stopword :**")
        stopword
        st.header("Term Frequency :")
        df_tf
        st.header("Latent Dirichlet Allocation :")
        hasil_lda
