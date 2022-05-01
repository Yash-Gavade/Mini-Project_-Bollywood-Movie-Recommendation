import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
from wordcloud import WordCloud, STOPWORDS 
import textwrap
import string
import warnings
warnings.simplefilter('ignore')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# RECOMMENDATION SYSTEM USING COSINE SIMILARITY


imdb_movies = pd.read_csv('movies.csv')

data_recsys=imdb_movies[['original_title', 'genre', 'director', 'actors',
                         'description','writer',
                         'reviews_from_critics','reviews_from_users'
                         ]].reset_index(drop = True)

data_recsys.set_index('original_title', inplace = True)

data_recsys['genre'] = data_recsys['genre'].fillna('').astype('str').str.lower()
data_recsys['genre'] = data_recsys['genre'].str.split(',')

data_recsys['director'] = data_recsys['director'].fillna('').astype('str').str.lower()
data_recsys['director'] = data_recsys['director'].str.split(',')

data_recsys['actors'] = data_recsys['actors'].fillna('').astype('str').str.lower()
data_recsys['actors'] = data_recsys['actors'].str.split(',')

data_recsys['writer'] = data_recsys['writer'].fillna('').astype('str').str.lower()
data_recsys['writer'] = data_recsys['writer'].str.split(',')

data_recsys['reviews_from_critics'] = data_recsys['reviews_from_critics'].fillna('').astype('str').str.lower()
data_recsys['reviews_from_critics'] = data_recsys['reviews_from_critics'].str.split(',')

data_recsys['reviews_from_users'] = data_recsys['reviews_from_users'].fillna('').astype('str').str.lower()
data_recsys['reviews_from_users'] = data_recsys['reviews_from_users'].str.split(',')

data_recsys['description'] = data_recsys['description'].fillna('').astype('str').str.lower()
data_recsys['description'] = data_recsys['description'].str.translate(str.maketrans('', '', string.punctuation))

listStopwords = set(stopwords.words('english'))
filtered = []
ps = PorterStemmer() 
for i, text in enumerate(data_recsys['description'].str.split()):
    for word in text:

        if word not in listStopwords:

            word_stemmed = ps.stem(word)
            filtered.append(word_stemmed)
    data_recsys['description'][i] = filtered
    filtered = []


data_recsys['final_content'] = ''
for i, text in data_recsys.iterrows():
    words = ''
    for col in data_recsys.columns:
        words = words + ' '.join(text[col]) + ' '
    data_recsys['final_content'][i] = words

count = CountVectorizer()
count_matrix = count.fit_transform(data_recsys['final_content']).astype(np.uint8)

chunk_size = 500 
matrix_len = count_matrix.shape[0] 

def similarity_cosine(start, end):
    if end > matrix_len:
        end = matrix_len
    return cosine_similarity(X=count_matrix[start:end], Y=count_matrix)
cosine_similarity_all = []
i=0

for chunk_start in range(0, matrix_len, chunk_size):
    
    if i == 0: 
        cosine_sim = similarity_cosine(chunk_start, chunk_start+chunk_size)
    
    else :
        cosine_similarity_chunk= similarity_cosine(chunk_start, chunk_start+chunk_size)
        cosine_sim = np.concatenate((cosine_sim.astype(np.float32), cosine_similarity_chunk.astype(np.float32)))
    
    i= 1

index_movies = pd.Series(data_recsys.index)


def get_movies(title, cosine_sim = cosine_sim):
    recommended_movies = []
    index_movie_input = index_movies[index_movies == title].index[0]
    score_movies = pd.Series(cosine_sim[index_movie_input]).sort_values(ascending = False)
    top_10_index_movies = list(score_movies.iloc[1:11].index)
    for i in top_10_index_movies:
        recommended_movies.append(imdb_movies['original_title'].iloc[i] + ' (' + str(imdb_movies['year'].iloc[i]) + ')')
    return recommended_movies

get_movies('')