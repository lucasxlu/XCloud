"""
Database toolkits
"""
import sys
import datetime
import time

import numpy as np
import pymysql
import requests


def connect_mysql_db(host='localhost', username='lucasxu', passwd='lucasxu', db='xcloud'):
    """
    connect to mysql database
    :param host:
    :param name:
    :param passwd:
    :param db:
    :return:
    """
    conn = pymysql.connect(host=host, user=username, password=passwd, db=db, use_unicode=True, charset="utf8")

    return conn


def insert_to_api(conn, username, api_name, api_elapse, api_call_datetime, terminal_type, img_path, skin_disease):
    """
    insert a record to API TABLE
    :param conn:
    :return:
    """
    cursor = conn.cursor()
    sql = "INSERT INTO api(username, api_name, api_elapse, api_call_datetime, terminal_type, img_path, skin_disease) " \
          "VALUES('%s', '%s', '%f', '%s', '%s', '%s', '%s')" % (
              username, api_name, api_elapse, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
              terminal_type, img_path, skin_disease)
    try:
        cursor.execute(sql)
        conn.commit()
    except:
        print('Error occurs while inserting to api TABLE!')


def insert_to_user(conn, username, register_datetime, register_type, user_organization, email, userkey, password):
    """
    insert a record to USER TABLE
    :param conn:
    :return:
    """
    cursor = conn.cursor()
    sql = "INSERT INTO users(username, register_datetime, register_type, user_organization, email, userkey, password) " \
          "VALUES('%s', '%s', %d, '%s', '%s', '%s', '%s')" % (
              username, str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))), register_type,
              user_organization, email, userkey, password)
    try:
        cursor.execute(sql)
        conn.commit()
    except:
        print('Error occurs while inserting to user TABLE!')
        pass


def query_api_hist(conn, username):
    """
    query API call details via hist
    :param conn:
    :param username:
    :return:
    """
    cursor = conn.cursor()
    sql = "SELECT api_name, COUNT(api_name) FROM api WHERE 1=1"
    if username is not None and username != '':
        sql += " AND username='{0}'".format(username)
    cursor.execute(sql)
    results = cursor.fetchall()

    db_result = {}
    for k, v in results:
        db_result[k] = v

    conn.close()

    return db_result


if __name__ == '__main__':
    conn = connect_mysql_db()
    # print(query_api_hist(conn, 'BigBear'))
    # insert_to_api(conn, 'BigBear', 'cv/fbp', 0.31, datetime.time(), 1)
    insert_to_user(conn, 'SmallBear', datetime.time(), 0, 'Tsinghua University', 'smallbear@thu.edu.cn', 'smallbear',
                   '66666a')
