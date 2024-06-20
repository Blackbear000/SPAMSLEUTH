
def filter_redundant_groups(identified_spam_groups):
    # 初始化一个列表用于存放最终的垃圾组
    filtered_groups = []

    # 将每个组的成员转换为集合，方便比较
    group_sets = [(group['Restaurant_id'], set(group['Spammer_group']), group) for group in identified_spam_groups]

    # 检查每个组是否是其他任何组的子集
    for i, (rest_id_i, members_i, group_i) in enumerate(group_sets):
        is_subset = False
        for j, (rest_id_j, members_j, group_j) in enumerate(group_sets):
            if i != j and rest_id_i == rest_id_j and members_i < members_j:
                is_subset = True
                break
        if not is_subset:
            filtered_groups.append(group_i)

    return filtered_groups

def print_spam_groups_info(identified_spam_groups):
    if not identified_spam_groups:
        print("No spam groups identified.")
        return
    identified_spam_groups = filter_redundant_groups(identified_spam_groups)
    spam_group_details = []
    for group in identified_spam_groups:
        restaurant_id = float(group['Restaurant_id'])
        confidence_level = group['K_metric']
        members = group['Spammer_group']
        # 将成员列表中的 int64 类型数据转换为 Python 内置的 int 类型
        members = [int(member) for member in members]
        spam_group_details.append({
            'Restaurant_id': restaurant_id,
            'Confidence_level': confidence_level,
            'Members': members
        })

    # for detail in spam_group_details:
    #     print(f"Restaurant ID: {detail['Restaurant_id']}, Type: {type(detail['Restaurant_id'])}")
    #     print(f"Confidence Level: {detail['Confidence_level']}, Type: {type(detail['Confidence_level'])}")
    #     print("Members:")
    #     for member in detail['Members']:
    #         print(f"{member}, Type: {type(member)}")
    #     print()

    return spam_group_details



