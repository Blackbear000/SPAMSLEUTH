from data_loading import load_data
import warnings
from evaluation import evaluate_all_restaurants, set_global_confidence_threshold_and_identify_spam_groups
from visualization import visualize_restaurant_and_reviewers, visualize_spam_group_global_relationships
from group_cal import print_spam_groups_info
warnings.filterwarnings('ignore')
def main():
    filepath = 'E:\\NUS2\\GP2\\GP2\\dataset_c.csv'
    data = load_data(filepath)
    ks = range(2, 15)
    all_results = evaluate_all_restaurants(data, ks)
    identified_spam_groups = set_global_confidence_threshold_and_identify_spam_groups(all_results, percentile=90)
    print_spam_groups_info(identified_spam_groups)

    restaurant_id1 = 4
    restaurant_id2 = 79


    # 特定餐厅的评论者网络
    visualize_restaurant_and_reviewers(data, identified_spam_groups, restaurant_id1)
    # 特定餐厅的垃圾组成员在全局范围内的评论关系
    visualize_spam_group_global_relationships(data, identified_spam_groups, restaurant_id2)
if __name__ == "__main__":
    main()
