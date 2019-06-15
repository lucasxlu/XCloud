"""
Web crawlers for Maimai
"""
import random
import time
import requests
from pymongo import MongoClient

COOKIES_STRING = ''


def conn_to_mongo(collection_name):
    """
    connect to MongoDB
    :return:
    """
    client = MongoClient()
    if collection_name == 'zhiyan':
        db = client.maimai.zhiyan
    elif collection_name == 'zhiyancomments':
        db = client.maimai.zhiyancomments

    return db


def insert_into_mongodb(db, collection_name, obj):
    """
    insert into MongoDB
    :param db:
    :param collection_name:
    :param obj:
    :return:
    """
    if db is None:
        db = conn_to_mongo(collection_name)
    db.insert_one(obj)


def crawl_maimai_zhiyan(company_name):
    """
    crawl maimai zhiyan
    :param company_name:
    :return:
    """
    url = 'https://maimai.cn/search/gossips'
    offset = 0
    continue_crawl = True
    cookies = dict(cookies_are=COOKIES_STRING)

    print('Start crawling Zhiyan for {}.'.format(company_name))

    while continue_crawl:
        payload = {
            'query': company_name,
            'limit': '20',
            'offset': offset,
            'searchTokens': [],
            'highlight': 'true',
            'sortby': 'time',
            'jsononly': 1,
        }

        response = requests.get(url=url, params=payload, headers={
            # ':authority': 'maimai.cn',
            # ':method': 'GET',
            # ':scheme': 'https',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome'
                          '/73.0.3683.75 Safari/537.36'
        }, cookies=cookies)

        if response.status_code == 200:
            data = response.json()
            if len(data['data']['gossips']) == 0:
                continue_crawl = False
            else:
                print(data)

                collection_name = 'zhiyan'
                db = conn_to_mongo(collection_name)

                for gossip in data['data']['gossips']:
                    gossip['company_name'] = company_name
                    insert_into_mongodb(db, collection_name, gossip)

                offset += 20

        time.sleep(random.randint(2, 5))  # a range between 2s and 5s


if __name__ == '__main__':
    # crawl_maimai_zhiyan('京东')
    db = conn_to_mongo('zhiyan')
    gossips = db.find({"company_name": "京东"})

    gossip_texts = []
    for gossip in gossips:
        gossip_texts.append(gossip['gossip']['text'])

    print(''.join(gossip_texts))
