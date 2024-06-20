import numpy as np
import warnings
from SpammerGroupDetection.data_loading import get_CandidateGroupsID
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def text_similarity(df, Product_id):
    datapool, ID = get_CandidateGroupsID(df, Product_id=Product_id)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(datapool['Review_Text'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    marks = cosine_sim.sum(axis=1)
    return ID, marks

def Rating_similarity(Product_id):
    pd, ID = get_CandidateGroupsID(Product_id=Product_id)

    spam_data = pd[pd['Label'] == -1]
    non_spam_data = pd[pd['Label'] == 0]

    rating_differences = []
    for _, row in pd.iterrows():
        user_rating = row['Rating']
        diff_spam = spam_data['Rating'].apply(lambda x: abs(user_rating - x)).sum()
        diff_non_spam = non_spam_data['Rating'].apply(lambda x: abs(user_rating - x)).sum()
        rating_difference = diff_spam - diff_non_spam
        rating_differences.append(rating_difference)

    return ID, rating_differences

def sentiment_difference(Product_id):
    pd, ID = get_CandidateGroupsID(Product_id=Product_id)

    spam_data = pd[pd['Label'] == -1]
    non_spam_data = pd[pd['Label'] == 1]

    polarity_diffs = []
    subjectivity_diffs = []

    for _, row in pd.iterrows():
        # Polarity
        polarity_spam_diff = spam_data['Sentiment_Polarity'].apply(lambda x: abs(row['Sentiment_Polarity'] - x)).sum()
        polarity_non_spam_diff = non_spam_data['Sentiment_Polarity'].apply(
            lambda x: abs(row['Sentiment_Polarity'] - x)).sum()
        net_polarity_diff = polarity_spam_diff - polarity_non_spam_diff
        polarity_diffs.append(net_polarity_diff)

        # subject
        subjectivity_spam_diff = spam_data['Sentiment_Subjectivity'].apply(
            lambda x: abs(row['Sentiment_Subjectivity'] - x)).sum()
        subjectivity_non_spam_diff = non_spam_data['Sentiment_Subjectivity'].apply(
            lambda x: abs(row['Sentiment_Subjectivity'] - x)).sum()
        net_subjectivity_diff = subjectivity_spam_diff - subjectivity_non_spam_diff
        subjectivity_diffs.append(net_subjectivity_diff)

    return ID, polarity_diffs, subjectivity_diffs


def calculate_topic_diversity(texts, n_components=5):
    """计算文本的主题多样性，适应文档数量较少的情况"""
    if len(texts) < 2:
        return 0
    vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_components, random_state=0)
    lda_output = lda.fit_transform(dtm)

    # 计算每个文档的主题分布的熵，熵越高表示主题分布越均匀，多样性越高
    entropy = np.apply_along_axis(lambda x: -np.sum(x * np.log(x + 1e-10)), 1, lda_output)
    return np.mean(entropy)

def user_topic_diversity(Product_id):
    pd, ID = get_CandidateGroupsID(Product_id=Product_id)
    users = pd['Review_id'].unique()
    data=load_data()
    diversity_scores = []

    for user in users:
        user_reviews = data[data['Review_id'] == user]['Review_Text'].tolist()
        if user_reviews and len(user_reviews) >= 2:  # 确保用户有足够的评论
            diversity_score = calculate_topic_diversity(user_reviews)
            diversity_scores.append(1 / diversity_score)
        else:
            diversity_scores.append(1e-4)

    return ID, diversity_scores

def count_spam_comments(Product_id):
    pd, ID = get_CandidateGroupsID(Product_id=Product_id)
    users = pd['Review_id'].unique()
    spam_comment_counts = []

    for user in users:
        user_spam_comments = pd[(pd['Review_id'] == user) & (pd['Label'] == -1)]
        spam_comment_counts.append(len(user_spam_comments))
    return ID, spam_comment_counts


def max_monthly_comment_frequency(Product_id):
    warnings.filterwarnings("ignore")  # 忽略警告
    pd, ID = get_CandidateGroupsID(Product_id=Product_id)

    pd['Review_Date'] = pd.to_datetime(pd['Review_Date'], format='%d/%m/%Y')
    pd['YearMonth'] = pd['Review_Date'].dt.to_period('M')
    monthly_comments = pd.groupby(['Review_id', 'YearMonth']).size()
    max_comment_frequencies = monthly_comments.groupby(level=0).max().tolist()

    return ID, max_comment_frequencies
