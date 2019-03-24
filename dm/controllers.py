import json
import os
import time

from django.http import HttpResponse
from pymongo import MongoClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def conn_to_mongo(db_name="haodf"):
    """
    connect to MongoDB
    :return:
    """
    client = MongoClient()
    if db_name == 'haodf':
        db = client.haodf.detail
    elif db_name == 'sanjiu':
        db = client.sanjiu.pifuke

    return db


def get_skin_info_from_kb_by_name(request):
    """
    get skin detailed information from knowledge base by disease name
    :param request:
    :return:
    """
    result = {'code': 0, 'msg': 'success'}

    tik = time.time()

    try:
        skin_disease_name = request.GET.get('skinDiseaseName').strip()
        db = conn_to_mongo('sanjiu')
        datalist = []
        docs = db.find({"疾病名称": skin_disease_name} if skin_disease_name != "" else {})
        for doc in docs:
            data = {}
            for k, v in doc.items():
                if k == "_id":
                    data[k] = str(v).replace('ObjectId("', '').replace('")', '')
                else:
                    data[k] = v
            datalist.append(data)
        result['results'] = datalist
    except:
        result['results'] = None

    result['elapse'] = time.time() - tik
    json_result = json.dumps(result, ensure_ascii=False)

    return HttpResponse(json_result)
