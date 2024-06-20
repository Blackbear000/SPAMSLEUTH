from textblob import TextBlob
from transformers import pipeline
import pandas as pd
import os


def login_huggingface():
    token = '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'
    command = f"huggingface-cli login --token {token}"
    os.system(command)
    return 1


def analyze_sentiment(text):
    """Analyzes the sentiment of a given text using TextBlob and returns the polarity and subjectivity."""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity


def format_text_based_on_sentiment(text):
    """Formats text based on sentiment analysis results with specific conditions."""
    polarity, subjectivity = analyze_sentiment(text)
    if polarity < 0.1 or subjectivity > 0.60:
        formatted_text = f"### Question: {text}, Sentiment_Polarity(-1 to 1) is {polarity}, Sentiment_Subjectivity(0 to 1) is {subjectivity}, rating is -1"
    else:
        formatted_text = text
    return formatted_text


def text_input(text):
    status = format_text_based_on_sentiment(text)
    # print(status)
    return status


def csv_input(path):
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    list = []
    for review in df['Review_Text']:
        list.append(format_text_based_on_sentiment(review))
        # print(format_text_based_on_sentiment(review))
    return list


def prepare_pipe():
    pipe = pipeline("text-classification", model="syannn/llama-2-7b-spammer")
    return pipe


def spam_detect(pipe, text):
    status = pipe(text)[0]
    return status


def main():
    # csv_input('dataset_case.csv')
    # os.system("huggingface-cli cache clear")
    # os.environ['HF_HOME'] = "F:/huggingface_cache"
    login_huggingface()
    
    text = '''
    Terrible experience! The lasagna was overcooked, and the sauce tasted like it came from a jar. Save your money and eat somewhere else.    '''
    formatted_text = text_input(text)
    pipe = prepare_pipe()
    output = spam_detect(pipe=pipe, text=formatted_text)
    print(output)
    # list=csv_input('E:\\NUS2\GP2\GP2\dataset_case.csv')
    #prepare_pipe一步第一次需要加载huggingface的模型文件，很大。可以考虑拆分出来提前下载。
    # pipe = prepare_pipe()
    # for texts in list:
    #     formatted_text=text_input(texts)
    #     output = spam_detect(pipe=pipe, text=formatted_text)
    #     print(output)

# !pip install transformers datasets evaluate accelerate
# main()
