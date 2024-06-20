import numpy as np
import pandas as pd
from SpammerGroupDetection.analytics import text_similarity, user_topic_diversity, sentiment_difference, Rating_similarity, \
    count_spam_comments, max_monthly_comment_frequency
from SpammerGroupDetection.data_loading import get_CandidateGroupsID
from tqdm import tqdm

def normalize_scores_using_cdf(data):
    for column in data.columns:
        if column not in ['Review_id', 'Product_id']:
            data[column] = data[column].rank(method='average', pct=True)
    return data


def compute_overall_score(data):
    data_squared = data.apply(lambda x: x ** 2 if x.name not in ['Review_id', 'Product_id'] else x)
    data['combined_score'] = np.sqrt(data_squared.loc[:, data_squared.columns != 'Review_id'].mean(axis=1))
    return data


def normalize_and_combine_scores(df, Product_id):
    ID0, tsi = text_similarity(df, Product_id)
    ID1, rsi = Rating_similarity(df, Product_id)
    ID2, pdi, sdi = sentiment_difference(df, Product_id)
    ID3, tdi = user_topic_diversity(df, Product_id)
    ID4, sci = count_spam_comments(df, Product_id)
    ID5, mcf = max_monthly_comment_frequency(df, Product_id)

    data = pd.DataFrame({
        'Product_id': Product_id,
        'Review_id': ID1,
        'text_similarity': tsi,
        'rating_similarity': rsi,
        'polarity_diffs': pdi,
        'subjectivity_diffs': sdi,
        'topic_diversity': tdi,
        'spam_count': sci,
        'max_frequency': mcf,
    })
    return compute_overall_score(normalize_scores_using_cdf(data))


def evaluate_spammer_group(data, restaurant_id, K):
    restaurant_data = data[data['Product_id'] == restaurant_id]
    normalized_data = normalize_scores_using_cdf(restaurant_data)
    scored_data = compute_overall_score(normalized_data)
    top_k_users = scored_data.nlargest(K, 'combined_score')['Review_id'].tolist()

    all_comments = data[data['Review_id'].isin(top_k_users)]
    spam_user_ids = all_comments[all_comments['Label'] == -1]['Review_id'].unique().astype(float)

    k_metric = len(spam_user_ids) / K if K > 0 else 0

    return {
        'Restaurant_id': restaurant_id,
        'Spammer_group': top_k_users,
        'K_metric': k_metric,
    }


def evaluate_all_restaurants(data, ks):
    restaurant_ids = data['Product_id'].unique().astype(float)
    results = {k: [] for k in ks}

    for k in tqdm(ks, desc='Overall K Evaluations'):
        k_results = []
        for restaurant_id in restaurant_ids:
            result = evaluate_spammer_group(data, restaurant_id, k)
            result['K_metric'] = result['K_metric']
            k_results.append(result)
        results[k] = k_results
    return results


def set_global_confidence_threshold_and_identify_spam_groups(results, percentile):
    all_k_metrics = [result['K_metric'] for k_results in results.values() for result in k_results]
    confidence_threshold = np.percentile(all_k_metrics, percentile)
    print(f"Global confidence threshold (percentile={percentile}): {confidence_threshold}")

    identified_spam_groups = []
    for k_results in results.values():
        for result in k_results:
            if result['K_metric'] > confidence_threshold:
                identified_spam_groups.append(result)
    return identified_spam_groups
