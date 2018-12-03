import json
from collections import OrderedDict

import jieba as jieba
import jieba.analyse
from django.http import HttpResponse
from django.shortcuts import render

USER_DICT = './userdict.txt'
STOPWORDS_FILE = './stopwords.txt'


# Create your views here.

def welcome(request):
    """
    welcome page for NLP welcome
    :param request:
    :return:
    """
    return render(request, 'welcome.html')


def word_seg(request):
    """
    sentence segmentation by jieba
    @Note: supported by jieba
    :param request: sentence
    :return:
    """
    sentence = request.GET.get('sentence')
    result = OrderedDict()

    if sentence is None:
        result['code'] = 1
        result['msg'] = 'Invalid Sentence Input'
        result['data'] = None
    else:
        result['code'] = 0
        result['msg'] = 'success'

        # jieba.load_userdict(USER_DICT)
        # jieba.analyse.set_stop_words(STOPWORDS_FILE)

        tags = jieba.analyse.extract_tags(sentence, topK=30, withWeight=True)

        seg_words = []
        tfidf = []

        for tag in tags:
            seg_words.append(tag[0])
            tfidf.append(tag[1])

        result['data'] = {
            'words': seg_words,
            'tf-idf': tfidf
        }

    json_result = json.dumps(result, ensure_ascii=False)

    return HttpResponse(json_result)


def sentiment(request):
    """
    calculate sentiment of a specific sentence
    @Note: currently supported by snownlp, but will be replaced customized model in the near future!
    :param request:
    :return:
    """
    sentence = request.GET.get('sentence')
    result = OrderedDict()

    if sentence is None:
        result['code'] = 1
        result['msg'] = 'Invalid Sentence Input'
        result['data'] = None
    else:
        result['code'] = 0
        result['msg'] = 'success'

        from snownlp import SnowNLP

        s = SnowNLP(sentence)

        senti_score = s.sentiments
        if senti_score >= 0.65:
            tp = 'positive'
        elif senti_score <= 0.4:
            tp = 'negative'
        else:
            tp = 'neutral'

        result['data'] = {
            'type': tp,
            'score': senti_score
        }

    json_result = json.dumps(result, ensure_ascii=False)

    return HttpResponse(json_result)
