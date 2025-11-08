import pandas as pd
import numpy as np
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from textstat import flesch_reading_ease
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords                                           
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
sia = SentimentIntensityAnalyzer()
CSV_PATH = r"C:\Users\CHAITANYA THAKUR\Downloads\chatgpt_chats_clean.csv"
def load_csv(uploaded_path=None, default_path=CSV_PATH):
    """
    Load uploaded CSV if provided, else load default CSV.
    """
    path_to_use = uploaded_path if uploaded_path and os.path.exists(uploaded_path) else default_path

    if not os.path.exists(path_to_use):
        raise FileNotFoundError(f"CSV not found at: {path_to_use}")

    df = pd.read_csv(path_to_use)
    expected_cols = ['Conversation_Title', 'Role', 'Message']
    for col in expected_cols:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")

    print(f"Loaded CSV: {path_to_use}")
    return df.head(1000)


def run_sentiment(df):
    assistant_msgs = df[df['Role'].str.lower() == 'assistant'].copy()
    if assistant_msgs.empty:
        return df
    assistant_msgs['Sentiment_Score'] = assistant_msgs['Message'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    def label_sentiment(score):
        if score >= 0.05: return 'Positive'
        if score <= -0.05: return 'Negative'
        return 'Neutral'
    assistant_msgs['Sentiment_Label'] = assistant_msgs['Sentiment_Score'].apply(label_sentiment)
    X = assistant_msgs['Sentiment_Score'].values.reshape(-1, 1)
    le = LabelEncoder()
    y = le.fit_transform(assistant_msgs['Sentiment_Label'])
    if len(X) < 10 or len(np.unique(y)) < 2:
        return assistant_msgs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {svm_acc:.2f}")
    knn_model = KNeighborsClassifier(n_neighbors=min(5, max(1, len(X_train)//5)))
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    print(f"KNN Accuracy: {knn_acc:.2f}")
    try:
        probs = svm_model.predict_proba(X)
        pos_idx = 0
        for idx, cls in enumerate(le.classes_):
            if cls.lower() == 'positive':
                pos_idx = idx
                break
        assistant_msgs['SVM_Pos_Prob'] = probs[:, pos_idx]
    except Exception:
        assistant_msgs['SVM_Pos_Prob'] = np.nan
    assistant_msgs['KNN_Pred_Label'] = le.inverse_transform(knn_model.predict(X))
    return assistant_msgs

def run_topics(df, n_topics=5):
    stop_words = set(stopwords.words('english'))
    texts = df['Message'].astype(str).tolist()
    clean = [" ".join([w for w in t.lower().split() if w.isalpha() and w not in stop_words]) for t in texts]
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=2000)
    dtm = vectorizer.fit_transform(clean)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topics = [{'topic_id': i, 'terms': [feature_names[j] for j in comp.argsort()[:-11:-1]]} for i, comp in enumerate(lda.components_)]
    try:
        dense = dtm.toarray()
    except Exception:
        dense = dtm.todense()
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(dense)
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    df['Cluster_KMeans'] = kmeans.fit_predict(reduced)
    hclust = AgglomerativeClustering(n_clusters=n_topics)
    df['Cluster_HC'] = hclust.fit_predict(reduced)
    try:
        knn_for_kmeans = KNeighborsClassifier(n_neighbors=5)
        knn_for_kmeans.fit(reduced, df['Cluster_KMeans'])
        df.attrs['knn_for_kmeans'] = knn_for_kmeans
    except Exception:
        pass
    print(f"Topics extracted & clustered into {n_topics} groups.")
    return topics, df

def compute_quality(df):
    df2 = df.copy()
    df2['Sentiment_Score'] = df2.get('Sentiment_Score', 0)
    df2['response_length'] = df2['Message'].apply(lambda x: len(str(x).split()))
    df2['readability'] = df2['Message'].apply(lambda x: flesch_reading_ease(str(x)))
    scaler = MinMaxScaler()
    features = df2[['response_length', 'readability', 'Sentiment_Score']].fillna(0)
    scaled = scaler.fit_transform(features)
    weights = np.array([0.2, 0.3, 0.5])
    df2['quality_score'] = np.tanh(np.dot(scaled, weights)) * 5
    df2['quality_score'] = df2['quality_score'].round(2)
    try:
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(scaled)
        df2['quality_PC1'] = pcs[:, 0]
        df2['quality_PC2'] = pcs[:, 1]
    except Exception:
        df2['quality_PC1'] = np.nan
        df2['quality_PC2'] = np.nan
    print("Quality scores computed.")
    return df2

def compute_repetition(df):
    messages = df['Message'].astype(str).tolist()
    if len(messages) < 2:
        return df
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X = vectorizer.fit_transform(messages)
    sims = cosine_similarity(X)
    repetition_score = np.mean(sims)
    df['Semantic_Repetition'] = repetition_score
    print(f"Average semantic repetition score: {repetition_score:.3f}")
    try:
        dense = X.toarray()
        pca = PCA(n_components=min(10, dense.shape[1]), random_state=42)
        red = pca.fit_transform(dense)
        nn = NearestNeighbors(n_neighbors=min(5, len(red)-1)).fit(red)
        distances, indices = nn.kneighbors(red)
        avg_neighbor_dist = distances.mean(axis=1)
        df['Avg_NN_Dist_PCA'] = avg_neighbor_dist
    except Exception:
        df['Avg_NN_Dist_PCA'] = np.nan
    return df

def find_similar_responses(df, query, k=3):
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X = vectorizer.fit_transform(df['Message'].astype(str))
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X)[0]
    top_k = np.argsort(sims)[-k:][::-1]
    results = df.iloc[top_k][['Message', 'Role']].copy()
    results['Similarity'] = sims[top_k]
    try:
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        dist, idxs = nn.kneighbors(q_vec, n_neighbors=k)
        knn_sims = np.clip(1 - dist.flatten() / (np.max(dist) + 1e-9), 0, 1)
        results['KNN_Similarity'] = knn_sims
    except Exception:
        results['KNN_Similarity'] = np.nan
    return results

def compute_repetition_pairs(df_pairs):
    if 'Message1' not in df_pairs.columns or 'Message2' not in df_pairs.columns:
        raise KeyError("Expected columns: 'Message1' and 'Message2'")
    df_pairs['Message1'] = df_pairs['Message1'].fillna('').astype(str)
    df_pairs['Message2'] = df_pairs['Message2'].fillna('').astype(str)
    df_pairs = df_pairs[(df_pairs['Message1'].str.strip() != '') & (df_pairs['Message2'].str.strip() != '')]
    if df_pairs.empty:
        df_pairs['similarity_score'] = []
        df_pairs['Is_Repetitive'] = []
        return df_pairs
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    combined = df_pairs['Message1'].tolist() + df_pairs['Message2'].tolist()
    X = vectorizer.fit_transform(combined)
    half = len(df_pairs)
    X1, X2 = X[:half], X[half:]
    sims = np.array([cosine_similarity(X1[i], X2[i])[0][0] for i in range(half)])
    df_pairs['similarity_score'] = sims
    df_pairs['Is_Repetitive'] = (df_pairs['similarity_score'] > 0.5).astype(int)
    return df_pairs

def lda_on_sentiment_features(df):
    if 'Sentiment_Label' not in df.columns:
        raise KeyError("Dataframe must contain 'Sentiment_Label' to run LDA.")
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X = vectorizer.fit_transform(df['Message'].astype(str)).toarray()
    y = LabelEncoder().fit_transform(df['Sentiment_Label'])
    pca = PCA(n_components=min(50, X.shape[1]), random_state=42)
    X_red = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_red, y)
    transformed = lda.transform(X_red)
    print("LDA done — discriminant shape:", transformed.shape)
    return lda, transformed, pca, vectorizer
