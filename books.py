# Book Recommender System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
import seaborn as sns

import warnings; warnings.simplefilter('ignore')

md = pd.read_csv('books.csv')
md.head()
md.columns 

# Data Analysis
print(md['average_rating'].describe())
best = md.sort_values('average_rating',ascending=False).iloc[0]['title']
worst = md.sort_values('average_rating',ascending=True).iloc[0]['title']

print("Best Rated movie is: {}".format(best))
print("Worst Rated movie is: {}".format(worst))

sns.set(style="darkgrid")
sns.countplot(x="author", data=smd)

# Simple Recommender

vote_counts = md[md['ratings_count'].notnull()]['ratings_count'].astype(int)

vote_averages = md[md['average_rating'].notnull()]['average_rating'].astype(int)

C = vote_averages.mean()
C

m = vote_counts.quantile(0.95)
m

qualified = md[(md['ratings_count']>= m) & (md['ratings_count'].notnull()) & (md['average_rating'].notnull())][['title','original_publication_year','ratings_count','average_rating']] 

qualified['ratings_count'] = qualified['ratings_count'].astype(int)

qualified['average_rating'] = qualified['average_rating'].astype(int)

qualified.shape

def weighted_rating(b):
    v = b['ratings_count']
    R = b['average_rating']
    return (v/(v+m) * R) + (m/(m+v) * C)

qualified['wr'] = qualified.apply(weighted_rating,axis=1)
qualified = qualified.sort_values('wr',ascending=False)

qualified.head()

# We could classify according to genre
"""
s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1,drop=True)
s.name = 'genre'
gen_md = md.drop('genres',axis=1).join(s)

def build_chart(genre,percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['ratings_count'].notnull()]['ratings_count'].astype(int)
    vote_averages = df[df['average_rating'].notnull()]['average_rating'].astype(int)
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['ratings_count'] >= m) & (df['ratings_count'].notnull()) & (df['average_rating'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['ratings_count'].astype('int')
    qualified['vote_average'] = qualified['average_rating'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified
"""

# Content Based Filtering

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

smd = md.copy()

# I take the first author as author

def get_author(b):
    author = b['authors'].split(',')
    return author[0]
    
smd['author'] = smd.apply(get_author,axis=1)

author_list = smd['author'].value_counts()

smd_graph = author_list.head(5).reset_index()
sns.set(style="darkgrid")
sns.barplot(x="index",y="author",data=smd_graph)

# Popular Author 

print("Most Popular author is: {}".format(author_list.index[0]))

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['author'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

get_recommendations("Harry Potter and the Sorcerer's Stone (Harry Potter, #1)").head(10)
  
# Collaborative Filtering
reader = Reader()

ratings = pd.read_csv('ratings.csv')

ratings.head()

data = Dataset.load_from_df(ratings[['user_id','book_id','rating']],reader)
data.split(n_folds = 5)

svd = SVD()
evaluate(svd,data,measures=['RMSE','MAE'])

trainset = data.build_full_trainset()
svd.train(trainset)

ratings[ratings['user_id'] == 1]

svd.predict(1,302,3)
# For book with ID 302, we get an estimated prediction of 3.806
# It works on giving the rating according to how the other users have predicted the movie

# Hybrid Recommender

# Input - user_id and title
# Output - Similar movies

id_map = md.copy().set_index('title')
indices_map = id_map.set_index('work_id')

def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['work_id']
    #print(idx)
    book_id = id_map.loc[title]['book_id']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    book_indices = [i[0] for i in sim_scores]
    
    best_book = smd.iloc[book_indices][['title', 'ratings_count', 'average_rating', 'original_publication_year', 'work_id']]
    best_book['est'] = best_book['work_id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['book_id']).est)
    best_book = best_book.sort_values('est', ascending=False)
    return best_book.head(10)

hybrid(1,'Harry Potter and the Prisoner of Azkaban (Harry Potter, #3)')
                                                    
# Gave movie suggestions to a particular user based on the estimated ratings that it had internally calculated for that user                                                         

# Implementing kNN
data = pd.merge(md,ratings,on='book_id')
pivot = data.pivot(index='title',columns='user_id',values='rating').fillna(0)
from scipy.sparse import csr_matrix
matrix = csr_matrix(pivot.values)

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
model_knn.fit(matrix)

# Let's test the model to get recommendations
query_index = np.random.choice(pivot.shape[0])
distances,indices = model_knn.kneighbors(pivot.iloc[query_index,:].reshape(1,-1),n_neighbors=6)

for i in range(0,len(distances.flatten())):
    if i==0:
        print('Recommendations for {0}:\n'.format(pivot.index[query_index]))
    else:
        print('{0}: {1}'.format(i,pivot.index[indices.flatten()[i]]))
        
# Using Matrix Factorization (Not perfect... think so used when data amount is large)
pivot2 = data.pivot(index='user_id',columns='title',values='rating').fillna(0)
pivot2.head()

pivot2.shape

X = pivot2.values.T
X.shape

import sklearn 
from sklearn.decomposition import TruncatedSVD

SVD = TruncatedSVD(n_components=3,random_state=17)
svd_matrix = SVD.fit_transform(X)
svd_matrix.shape

# We use Pearson's R correlation coefficient for every book pair in our final matrix
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)
corr = np.corrcoef(matrix)
corr.shape

title = pivot2.columns
book_list = list(title)
coffey_hands = book_list.index("The Alchemist")
print(coffey_hands)

corr_coffey_hands = corr[coffey_hands]
list(title[(corr_coffey_hands<1.0) & (corr_coffey_hands>0.9)])
