import numpy as np
import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# LOADING DATASET
movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('dataset/tmdb_5000_credits.csv')

# print(movies.head())
# print(credits.head())

# MERGED movies AND credits DATASETS based on title
movies = movies.merge(credits, on='title')
# print(movies.shape)
# print(credits.shape)

# print(movies.head())
# print(movies.info())

# EXTRACTED ALL THE RELEVANT COLUMNS
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
# print(movies.isnull().sum())

# DROPPED ALL THE NULL VALUES
movies.dropna(inplace=True)

# print(movies.duplicated().sum())

# FUNCTION THAT TAKES DICTIONARY OF GENRES AND KEYWORDS AS INPUT 
# AND RETURNS A LIST OF NAMES OF GENRES AND KEYWORDS
def preprocess(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# PREPROCESSED GENRES COLUMN
movies['genres'] = movies['genres'].apply(preprocess)
# print(movies['genres'])

# PREPROCESS KEYWORDS COLUMN
movies['keywords'] = movies['keywords'].apply(preprocess)
# print(movies.head())

# print(movies['cast'][0])

# FUNCTION TO EXTRACT THE TOP 3 MOVIE CAST
def top3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(top3)
# print(movies['cast'])

# print(movies.head())

# print(movies['crew'][0])

# FUNCTION TO EXTRACT THE DIRECTOR OF A MOVIE
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# PREPROCESSED THE CREW COLUMN
movies['crew'] = movies['crew'].apply(fetch_director)
# print(movies.head())
# print(movies['crew'])
# print(movies['overview'][0])

# PREPROCESSED THE OVERVIEW COLUMN 
movies['overview'] = movies['overview'].apply(lambda x: x.split())
# print(movies.head())

# PREPROCESSED THE GENRES COLUMN
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(' ','') for i in x])
# print(movies['genres'])

movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(' ','') for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(' ','') for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(' ','') for i in x])

# print(movies.head())

# CONSTRUCTED A NEW COLUMN NAMED tags FROM GENRE, KEYWORDS, CAST AND CREW INFO 
# FROM THEIR RESPECTIVE COLUMNS
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
# print(movies.head())

# CONSTRUCTED A NEW DATAFRAME CONTAINING movie_id, title, tags OF MOVIES
new_df = movies[['movie_id','title','tags']]
# print(new_df.head())

new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x))
# print(new_df)

ps = PorterStemmer()

# FUNCTION TO STEM ALL THE WORDS  OF A GIVEN TEXT
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return ' '.join(y)

# APPLIED STEMMING TO tags COLUMN
new_df['tags'] = new_df['tags'].apply(stem)
# print(new_df['tags'][0])

# CONVERTED ALL THE LETTERS TO LOWERCASE
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
# print(new_df.head())

cv = CountVectorizer(max_features=5000, stop_words='english')
# VECTORIZED THE tags COLUMN
vectors = cv.fit_transform(new_df['tags']).toarray()

# CALCULATED THE COSINE SIMILARITY OF THE RESULTANT VECTORS
similarity = cosine_similarity(vectors)

# print(similarity)

# new_df[new_df['title'] == 'Batman Begins'].index[0]

# FUNCTION TO PRINT THE TOP 5 SIMILAR MOVIE NAMES
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, 
                        key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recommend('Batman Begins')

pickle.dump(new_df.to_dict(), open('model/movie_dict.pkl','wb'))
pickle.dump(similarity, open('model/similarity.pkl','wb'))