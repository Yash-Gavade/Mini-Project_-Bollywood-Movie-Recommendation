import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS 
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

imdb_movies = pd.read_csv('movies.csv')

data_recsys=imdb_movies[['Movie_id', 'Movie_Title', 'Released_Year', 'Duration',
                         'Genre','Rating',
                         'Votes','Director','Actor 1','Actor 2','Actor 3'
                         ]].reset_index(drop = True)


data_recsys.set_index('Movie_id', inplace = True)

data_recsys['Movie_Title'] = data_recsys['Movie_Title'].fillna('').astype('str').str.lower()
data_recsys['Movie_Title'] = data_recsys['Movie_Title'].str.split(',')

data_recsys['Released_Year'] = data_recsys['Released_Year'].fillna('').astype('str').str.lower()
data_recsys['Released_Year'] = data_recsys['Released_Year'].str.split(',')

data_recsys['Duration'] = data_recsys['Duration'].fillna('').astype('str').str.lower()
data_recsys['Duration'] = data_recsys['Duration'].str.split(',')

data_recsys['Genre'] = data_recsys['Genre'].fillna('').astype('str').str.lower()
data_recsys['Genre'] = data_recsys['Genre'].str.split(',')

data_recsys['Rating'] = data_recsys['Rating'].fillna('').astype('str').str.lower()
data_recsys['Rating'] = data_recsys['Rating'].str.split(',')

data_recsys['Votes'] = data_recsys['Votes'].fillna('').astype('str').str.lower()
data_recsys['Votes'] = data_recsys['Votes'].str.split(',')

data_recsys['Director'] = data_recsys['Director'].fillna('').astype('str').str.lower()
data_recsys['Director'] = data_recsys['Director'].str.translate(str.maketrans('', '', string.punctuation))

data_recsys['Actor 1'] = data_recsys['Actor 1'].fillna('').astype('str').str.lower()
data_recsys['Actor 1'] = data_recsys['Actor 1'].str.split(',')


data_recsys['Actor 2'] = data_recsys['Actor 2'].fillna('').astype('str').str.lower()
data_recsys['Actor 2'] = data_recsys['Actor 2'].str.split(',')


data_recsys['Actor 3'] = data_recsys['Actor 3'].fillna('').astype('str').str.lower()
data_recsys['Actor 3'] = data_recsys['Actor 3'].str.split(',')

listStopwords = set(stopwords.words('english'))
filtered = []
ps = PorterStemmer() 
for i, text in enumerate(data_recsys['Director'].str.split()):
    for word in text:
        # Filtering/Removing stopwords in the text
        if word not in listStopwords:
            # Stemming words
            word_stemmed = ps.stem(word)
            filtered.append(word_stemmed)
    data_recsys['Director'][i] = filtered
    filtered = []



count = CountVectorizer()
count_matrix = count.fit_transform(data_recsys['Director']).astype(np.uint8)

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

# Movies Recommendation function
def get_movies(title, cosine_sim = cosine_sim):
    recommended_movies = []
    index_movie_input = index_movies[index_movies == title].index[0]
    score_movies = pd.Series(cosine_sim[index_movie_input]).sort_values(ascending = False)
    top_10_index_movies = list(score_movies.iloc[1:11].index)
    for i in top_10_index_movies:
        recommended_movies.append(imdb_movies['Movie_Title'].iloc[i] + ' (' + str(imdb_movies['Released_Year'].iloc[i]) + ')')
    return recommended_movies

movie_input = str(input("Please enter the name of the movie.\n"))
get_movies(movie_input.capitalize())