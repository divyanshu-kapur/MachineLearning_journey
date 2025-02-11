#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Let's first explore the Fandango ratings to see if our analysis agrees with the article's conclusion.
fandango = pd.read_csv("C:\\Users\\DELL\\fandango_scrape.csv")

fandango.head()
fandango.info()
fandango.describe()

#Let's explore the relationship between popularity of a film and its rating using a scatter plot

plt.figure(figsize=(10,4),dpi=200)
sns.scatterplot(data=fandango,x='RATING',y='VOTES')
plt.title('Rating vs Votes')

# Lets check the correlation between the columns
fandango_1 = fandango.select_dtypes(exclude='object')
fandango_1.corr()

#new column that is able to strip the year from the title strings and set this new column as YEAR

fandango['YEAR'] = fandango['FILM'].apply(lambda title:title.split('(')[-1].strip(')') )

#visualize the count of movies per year
sns.countplot(fandango,x='YEAR')
fandango['YEAR'].value_counts()

#Top Ten movies with highest votes
fandango.nlargest(n=10, columns='VOTES')

#Top Ten movies with highest votes
fandango.nsmallest(n=10, columns='VOTES')

#Remove FILMs with zero votes
fandango_reviewed = fandango[fandango['VOTES'] > 0]
fandango_reviewed

plt.figure(figsize=(10,5),dpi=150)
sns.kdeplot(data=fandango_reviewed,x='RATING',fill=True,label='Rating',clip=[0,5])
sns.kdeplot(data=fandango_reviewed,x='STARS',fill=True,label='Stars',clip=[0,5])
plt.legend(loc=(1.05,0.5))

# Let's now actually quantify this discrepancy. Create a new column of the difference between STARS displayed versus true RATING.
fandango_reviewed.loc[:,'STARS DIFF'] = fandango_reviewed['STARS'] - fandango_reviewed['RATING']
fandango_reviewed['STARS DIFF'] = np.round(fandango_reviewed['STARS DIFF'],2)

fandango_reviewed.head()

# Lets visualize the same using countplot
plt.figure(figsize=(12,5),dpi=150)
sns.countplot(data=fandango_reviewed,x='STARS DIFF',palette='magma')

# We can see from the plot that one movie was displaying over a 1 star difference than its true rating! 
fandango_reviewed[fandango_reviewed['STARS DIFF'] == 1]

all_sites = pd.read_csv("C:\\Users\\DELL\\all_sites_scores.csv")

all_sites.head()
all_sites.info()
all_sites.describe()

#Rotten Tomatoes
plt.figure(figsize=(10,4),dpi=100)
sns.scatterplot(data=all_sites,x='RottenTomatoes',y='RottenTomatoes_User')

all_sites['RottenDiff'] = all_sites['RottenTomatoes'] - all_sites['RottenTomatoes_User']

all_sites['RottenDiff'].apply(abs).mean()

plt.figure(figsize=(8,4))
sns.histplot(all_sites,x='RottenDiff',kde=True,bins=20)
plt.title('RT Critics Score - RT User Score')

#With Absolute values
plt.figure(figsize=(8,4),dpi=150)
sns.histplot(x=all_sites['RottenDiff'].apply(abs),kde=True,bins=25)
plt.title('RT Critics Score - RT User Score')

all_sites.nsmallest(5,'RottenDiff')[['FILM', 'RottenDiff']]
all_sites.nlargest(5,'RottenDiff')[['FILM', 'RottenDiff']]

#Meta Critic
plt.figure(figsize=(10,4),dpi=150)
sns.scatterplot(data=all_sites,x='Metacritic',y='Metacritic_User',color='red')

#Meta Critic vote count vs IMDB vote counts
plt.figure(figsize=(10,4),dpi=150)
sns.scatterplot(data=all_sites,x='Metacritic_user_vote_count',y='IMDB_user_vote_count',color='green')

#movie with the highest IMDB user vote count
all_sites.nlargest(1,'IMDB_user_vote_count')[['FILM','IMDB_user_vote_count','Metacritic_user_vote_count']]

#movie with the highest Metacritic User user vote count
all_sites.nlargest(1,'Metacritic_user_vote_count')[['FILM','Metacritic_user_vote_count','IMDB_user_vote_count']]

# Fandango Scoers vs All Sites Scores

# Merge the datasets
df = pd.merge(fandango,all_sites,on='FILM',how='inner')

df.head()
df.info()
df.describe()

#Create new normalized columns for all ratings so they match up within the 0-5 star range shown on Fandango. 

df['RottenTomatoes_Norm'] = np.round(df['RottenTomatoes']/20,1)
df['RottenTomatoes_User_Norm'] = np.round(df['RottenTomatoes_User']/20,1)
df['Metacritic_Norm'] = np.round(df['Metacritic']/20,1)
df['Metacritic_User_Norm'] = np.round(df['Metacritic_User']/2,1)
df['IMDB_Norm'] = np.round(df['IMDB']/2,1)

df.head()
df.info()

#data with only normalized scores
df_norm = df[['STARS','RATING','RottenTomatoes_Norm','RottenTomatoes_User_Norm','Metacritic_Norm','Metacritic_User_Norm','IMDB_Norm']]

### Comparing Distribution of Scores Across Sites
def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)
fig, ax = plt.subplots(figsize=(15,6),dpi=150)
sns.kdeplot(data=df_norm,clip=[0,5],shade=True,palette='Set1',ax=ax)
move_legend(ax, "upper left")

plt.figure(figsize=(10,4),dpi=150)
sns.kdeplot(data = df_norm[['RottenTomatoes_Norm','STARS']],shade=True,clip=[0,5],palette='Set1')

plt.figure(figsize=(12,5),dpi=150)
sns.histplot(df_norm,bins=50)

sns.clustermap(df_norm)

df_norm_films = df[['FILM','STARS','RATING','RottenTomatoes_Norm','RottenTomatoes_User_Norm','Metacritic_Norm','Metacritic_User_Norm','IMDB_Norm']]

df_norm_films.nsmallest(10,'RottenTomatoes_Norm')

plt.figure(figsize=(12,5),dpi=150)
sns.kdeplot(df_norm.nsmallest(10,'RottenTomatoes_Norm'),shade=True,palette='Set1')
plt.title("Ratings for RT Critic's 10 Worst Reviewed Films")

### Fandango is showing around 3-4 star ratings for films that are clearly bad! Notice the biggest offender, Taken 3!. Fandango is displaying 4.5 stars on their site for a film with an average rating of 1.86 across the other platforms

df_norm_films.iloc[25]