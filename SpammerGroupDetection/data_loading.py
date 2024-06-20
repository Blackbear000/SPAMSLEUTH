import pandas as pd

def load_data(filepath):
    # filepath='/Users/liusiyan/PycharmProjects/spammer/spammer group detection/dataset_c.csv'
    data = pd.read_csv(filepath, encoding='ISO-8859-1')
    return data

def get_CandidateGroupsID(Product_id):
    df=pd.DataFrame(load_data())
    datapool = df[df['Product_id'] == Product_id]
    ID = datapool['Review_id'].tolist()
    return datapool, ID
