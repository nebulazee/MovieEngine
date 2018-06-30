
import pandas as pd
from nltk.metrics import edit_distance
def findmostSimilarItem(item):
    
	#Reading items file:
    i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
    encoding='latin-1')
    dist=0;min1=10000;title="";
    for i in range(1682):
        dist=edit_distance(item.lower(),items.iloc[i]['movie_title'].lower())
        if min1 > dist:
            min1=dist;
            title=items.iloc[i]['movie_title'];
    
    return title;
            
        