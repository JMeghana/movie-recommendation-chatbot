# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.parsing.preprocessing import remove_stopwords
import streamlit as st
pd.pandas.set_option('display.max_columns',None)

movies_data = pd.read_csv(r'C:\Users\j_meg\StreamLit_MJ\movie-recommendation-chatbot\data\movies_metadata.csv')
link_small = pd.read_csv(r'C:\Users\j_meg\StreamLit_MJ\movie-recommendation-chatbot\data\links_small.csv')
credits = pd.read_csv(r'C:\Users\j_meg\StreamLit_MJ\movie-recommendation-chatbot\data\credits.csv')
keyword = pd.read_csv(r'C:\Users\j_meg\StreamLit_MJ\movie-recommendation-chatbot\data\keywords.csv')

# Removing rows with the index labels 19730, 29503, and 35587
movies_data = movies_data.drop([19730, 29503, 35587])

# Convert the 'id' column to integers
movies_data['id'] = movies_data['id'].astype('int')

#Filtering 'link_small' DataFrame to get rows where 'tmdbId' is not null and converting 'tmdbId' to integers
link_small = link_small[link_small['tmdbId'].notnull()]['tmdbId'].astype('int')

#Creating 'smd' DataFrame by filtering 'movies_data' based on 'id' values present in 'link_small'
smd = movies_data[movies_data['id'].isin(link_small)]

#Filling missing values in 'tagline' and 'overview' columns with empty strings
smd['tagline'] = smd['tagline'].fillna('')
smd['overview'] = smd['overview'].fillna('')

keyword['id'] = keyword['id'].astype('int')
credits['id'] = credits['id'].astype('int')

movies_data_merged = movies_data.merge(keyword,on='id')
movies_data_merged = movies_data_merged.merge(credits,on='id')

smd2 = movies_data_merged[movies_data_merged['id'].isin(link_small)]

smd2['cast'] = smd2['cast'].apply(literal_eval)
smd2['crew'] = smd2['crew'].apply(literal_eval)
smd2['keywords'] = smd2['keywords'].apply(literal_eval)
smd2['cast_size'] = smd2['cast'].apply(lambda x: len(x))
smd2['crew_size'] = smd2['crew'].apply(lambda x: len(x))

# Extracting description from tagline and overview features

smd2['tagline'] = smd2['tagline'].fillna('').apply(lambda x: x.split(" "))
smd2['overview'] = smd2['overview'].fillna('').apply(lambda x: x.split(" "))
smd2["description"] = smd2['tagline'] + smd2['overview']

def get_director(names):
    for i in names:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

smd2['directors'] = smd2['crew'].apply(get_director)
smd2['directors'] = smd2['directors'].astype('str')

smd2['directors'] =smd2['directors'].apply(lambda x: [x,x, x])

smd2['cast'] = smd2['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd2['cast'] = smd2['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

smd2['keywords'] = smd2['keywords'].apply(lambda x: [ i['name'] for i in x] if isinstance(x,list) else [])

s = smd2.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level =1,drop = True)
s.name = 'keyword'
s = s.value_counts()
s = s[s>1]

def keywords(x):
    m = []
    for i in x:
        if i in s:
            m.append(i)
    return m

stemmer = SnowballStemmer('english')
smd2['keywords'] = smd2['keywords'].apply(keywords)

from itertools import chain
def process(x):
    it = [["".join(y) for y in i.split(" ")] for i in x]
    return list(chain.from_iterable(it))

smd2["keywords"] = smd2['keywords'].apply(process)

test1 = smd2.copy()
test1["soup"] = smd2['keywords'] + smd2['cast'] + smd2['directors'] + smd2["description"]
test1['soup'] = test1['soup'].apply(lambda x: " ".join(x))
test1['soup'] = test1['soup'].apply(lambda x: remove_stopwords(x))

stemmer = SnowballStemmer('english')
test1['soup'] = test1['soup'].apply(lambda x: x.split(" ")).apply(lambda x: [stemmer.stem(i) for i in x])
test1['soup'] = test1['soup'].apply(lambda x: " ".join(x))

d = {}
for item in ['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id',
       'imdb_id', 'original_language', 'original_title', 'overview',
       'popularity', 'poster_path', 'production_companies',
       'production_countries', 'release_date', 'revenue', 'runtime',
       'spoken_languages', 'status', 'tagline', 'title', 'video',
       'vote_average', 'vote_count', 'keywords', 'cast', 'crew', 'cast_size',
       'crew_size', 'description', 'directors', 'soup']:
    d[item] = "NA"

# ## **getPredictions Function**

def getPredictionsV2(soup, smd, num):
  smd = smd[smd.title != "myRequest"]

  # remove stopwords
  stopword_removed_soup = remove_stopwords(soup)

  # stem the input string:
  stemmer = SnowballStemmer('english')
  soup_list = stopword_removed_soup.split(" ")
  soup_list_stemmed = [stemmer.stem(i) for i in soup_list]
  stemmed_soup = " ".join(soup_list_stemmed)

  # print(stemmed_soup)

  d["soup"] = stemmed_soup

  d["title"] = "myRequest"
  smd = smd.append(d, ignore_index=True)

  tf = TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=0,stop_words='english')
  tfid_mat = tf.fit_transform(smd['soup'])
  cos_sim = linear_kernel(tfid_mat,tfid_mat)

  smd = smd.reset_index()
  titles = smd['title']
  indices = pd.Series(smd.index,index=smd['title'])

  idx = indices["myRequest"]#Gettting the index of the movie
  sim_scores = list(enumerate(cos_sim[idx])) #finding the cos similarity of the movie using its index and enumarating the similarity
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #sorting the movie based on the similarity score
  sim_scores = sim_scores[1:31] # taking first 30 movies
  movie_indices = [i[0] for i in sim_scores] # taking the sorted movies

  final_df = titles.iloc[movie_indices].head(num)
  final_df.index = range(1, len(final_df) + 1)
  
  # Use str.join() to concatenate index and movie names with the desired separator
  formatted_output = "\n".join(f"{i}. {movie}" for i, movie in enumerate(final_df, start=1))

  # return titles.iloc[movie_indices]
  return formatted_output

# Building streamlit app
st.title("Movie Recommendation Bot")
with st.chat_message("assistant"):
    st.write("""Welcome to our Movie Recommendation Chatbot!

I'm your friendly movie chatbot, and I'm here to help you discover your next favorite movies!

How it works:
1. You can input your movie preferences, genres, actors, directors, or any movie-related queries in the prompt.
2. Based on your input, I'll analyze our extensive movie database and provide you with a personalized list of the top 10 recommended movies that match your interests.

For example:\n
Prompt: Give me Christopher Nolan movies.\n
Recommended movies:
1. Following
2. The Prestige
3. Insomnia
4. Inception
5. Batman Begins
6. Interstellar
7. Memento
8. The Dark Knight
9. The Dark Knight Rises
10. Side by Side

Feel free to explore various movie queries, and I'll do my best to suggest exciting movie options for you! If you need help or have any specific movie preferences in mind, just let me know, and I'll be happy to assist.

Let the movie journey begin!""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Send a message"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = getPredictionsV2(prompt, test1, 10)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})