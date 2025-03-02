import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

file_path = "/Users/aizaz/Downloads/Books.csv" #update it according to your path
df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)



df  = df[['Book-Title', 'Book-Author', 'Publisher']].dropna()
df['combined_features'] = df['Book-Title'] + " " + df['Book-Author'] + " " + df['Publisher']


vectorize = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = vectorize.fit_transform(df['combined_features'])

nn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='auto')
nn.fit(tfidf_matrix)


def recommend_books(book_title):
    """Recommends simialr books based on a given title."""
    book_index = df[df['Book-Title'].str.lower() == book_title.lower()].index 
    if book_index.empty:
        return "Book not found in dataset"
    
    book_index = book_index[0]
    distances, indices = nn.kneighbors(tfidf_matrix[book_index], n_neighbors=6)
    
    recommended_books = df.iloc[indices[0][1:]]['Book-Title'].tolist()
    return recommended_books


book_name = "A Certain Justice"
print(f"Books simiar to '{book_name}':\n", recommend_books(book_name))