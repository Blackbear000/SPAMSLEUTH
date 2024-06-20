import networkx as nx
import matplotlib.pyplot as plt

import os

def visualize_restaurant_and_reviewers(app, data, identified_spam_groups, restaurant_id):
    save_dir = os.path.join(app.root_path, 'static', 'plots')
    os.makedirs(save_dir, exist_ok=True)
    G = nx.Graph()
    G.add_node(f"Restaurant {restaurant_id}", color='green', size=3000)  # 添加代表餐厅的节点

    spam_group_members = {}

    group_number = 1
    for group in identified_spam_groups:
        if group['Restaurant_id'] == restaurant_id:
            for member in group['Spammer_group']:
                if member not in spam_group_members:
                    spam_group_members[member] = []
                spam_group_members[member].append(str(group_number))
            group_number += 1

    all_spam_users = set(data[(data['Product_id'] == restaurant_id) & (data['Label'] == -1)]['Review_id'].tolist())

    restaurant_reviews = data[data['Product_id'] == restaurant_id]
    for index, row in restaurant_reviews.iterrows():
        user_node = f"User {row['Review_id']}"
        if row['Review_id'] in spam_group_members:
            node_color = 'red'
        elif row['Review_id'] in all_spam_users:
            node_color = 'orange'
        else:
            node_color = 'blue'

        G.add_node(user_node, color=node_color, size=300)

        group_tags = ', '.join(spam_group_members.get(row['Review_id'], []))
        G.add_edge(f"Restaurant {restaurant_id}", user_node, label=group_tags)

    colors = [G.nodes[n]['color'] for n in G.nodes]
    sizes = [G.nodes[n]['size'] for n in G.nodes]

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes)
    nx.draw_networkx_labels(G, pos, font_size=6)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title(f"Network of Restaurant {restaurant_id} and Reviewers")
    save_path = os.path.join(save_dir, f"restaurant_{restaurant_id}_reviewers.png")
    plt.savefig(save_path)
    plt.close()

def visualize_spam_group_global_relationships(app, data, identified_spam_groups, restaurant_id):
    save_dir = os.path.join(app.root_path, 'static', 'plots')
    os.makedirs(save_dir, exist_ok=True)
    G = nx.Graph()

    restaurant_node = f"Product {restaurant_id}"
    G.add_node(restaurant_node, color='green', size=3000)

    spam_group_members = {}
    group_number = 1
    for group in identified_spam_groups:
        if group['Restaurant_id'] == restaurant_id:
            for member in group['Spammer_group']:
                if member not in spam_group_members:
                    spam_group_members[member] = []
                spam_group_members[member].append(str(group_number))
            group_number += 1

    spam_comments = data[data['Review_id'].isin(spam_group_members)]

    for index, row in spam_comments.iterrows():
        user_node = f"User {row['Review_id']}"
        reviewed_restaurant_node = f"Product {row['Product_id']}"

        G.add_node(user_node, color='red', size=300)
        G.add_node(reviewed_restaurant_node, color='green', size=1000)

        edge_color = 'red' if row['Label'] == -1 else 'grey'
        edge_style = 'dashed' if row['Label'] == -1 else 'solid'

        edge_label = ', '.join(spam_group_members[row['Review_id']]) if row['Product_id'] == restaurant_id else None

        G.add_edge(user_node, reviewed_restaurant_node, color=edge_color, style=edge_style, label=edge_label)

    edge_colors = nx.get_edge_attributes(G, 'color').values()
    edge_styles = nx.get_edge_attributes(G, 'style').values()
    edge_labels = nx.get_edge_attributes(G, 'label')
    colors = [G.nodes[n]['color'] for n in G.nodes]
    sizes = [G.nodes[n]['size'] for n in G.nodes]
    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=sizes, edge_color=edge_colors, style=edge_styles)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)  # 使用小字体大小显示边的标签
    plt.title(f"Global Comment Relationships of Spam Group Members from Restaurant {restaurant_id}")
    save_path = os.path.join(save_dir, f"spam_group_relationships_restaurant_{restaurant_id}.png")
    plt.savefig(save_path)
    plt.close()