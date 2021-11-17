import pandas as pd 
import embeddinghub as eh

anime_list = pd.read_csv('anime.csv')

genre_df= anime_list["genre"].str.get_dummies(sep=",")
num_genres = len(genre_df.columns)

genre_df = pd.merge(anime_list[['anime_id','name']] , genre_df , left_index=True, right_index=True)

genre_df = genre_df.head(2000)

hub = eh.connect(eh.LocalConfig("data/"))

space = hub.create_space("anime", dims=num_genres)

emb = {}

for idx,anime in genre_df.iterrows():
    key = anime['name']
    embedding = anime.to_list()[2:]
    emb[key] = embedding

space.multiset(emb)

neighbors = space.nearest_neighbors(key="Kizumonogatari II: Nekketsu-hen", num=5)

print(
    anime_list[anime_list['name'] == 'Kizumonogatari II: Nekketsu-hen']['genre']
)
for neighbor in neighbors:
    print(neighbor)
    print(
    anime_list[anime_list['name'] == neighbor]['genre']
)