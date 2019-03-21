#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@163.com)
Date:    2018/8/10 15:31
"""
import pandas as pd
import numpy as np
from impala.dbapi import connect
from impala.util import as_pandas
from datetime import datetime
import os

impala_conn = connect(host='192.168.23.236', port=21050)
impala_cursor = impala_conn.cursor()
impala_cursor.execute('use odata')


def ai_response(param=None):
    if param is None:
        param = {'year': '2018',
                 'term_id': '2',
                 'subject_id': '',
                 'grade_id': '7',
                 'city_id': '020',
                 'level_id': '', }

    sql = """
          select 
            sb.class_level_id as level_id,
            sb.cuc_num,
            students_id as user_id,
            dim_student.stu_loginname,
            dim_student.stu_real_name,

            sb.knowledge_id,adk.knowledge_name,
            sb.problem_id as item_id, 
            sb.difficulty as difficulty,
            sb.discrimination,
            sb.new_difficulty,
            sb.system,
            isright as answer, 
            sb.ans_json,
            sb.create_time as date_time,
            sb.ans_time,
        rank() over(partition by students_id, sb.knowledge_id order by sb.create_time) as time_rank
        from ods_ai_study_ad_ans_behavior sb
        left join 
            (select knowledge_id,knowledge_name from odata.ods_ai_study_ad_knowledge group by knowledge_id,knowledge_name ) adk on adk.knowledge_id =sb.knowledge_id 
        left join dimdb.dim_student on dim_student.old_stu_id=sb.students_id
        where sb.city_id = '%(city_id)s'
        and sb.term_id = '%(term_id)s'
        and sb.year = 2018
        --and class_level_id = '%(level_id)s'
        -- and sb.subject_id = '%(subject_id)s'
        and sb.grade_id = '%(grade_id)s'
        """ % param
    today = datetime.today().strftime("%Y-%m-%d")
    file_name = "ai_response_%s.pkl" % today
    if os.path.exists(file_name):
        return pd.read_pickle(file_name)

    impala_cursor.execute(sql)
    stu_online = as_pandas(impala_cursor)
    stu_online.to_pickle(file_name)
    return stu_online


def aistudy_knowledge_question(param=None):
    if param is None:
        param = {'year': '2018', 'term_id': '2', 'subject_id': '', 'grade_id': '7', 'city_id': '020'}
    sql = """
        select 
          -- ad.cuc_num,
          -- cla.knowledge_id,
          -- cla.knowledge_name,ad
          ad.problem_id as item_id,
          ad.difficulty as difficulty,
          ad.discrimination,
          ad.new_difficulty
          -- ad.isdel
          --cla.id,
          -- cla.create_time
          -- rank() over(partition by cla.knowledge_id,cla.knowledge_name order by cla.id) as order_num
        from odata.ods_ai_study_ad_paper_question ad
       --join  odata.ods_ai_study_ad_cla_knowledge cla on ad.knowledge_id=cla.knowledge_id

    """ % param
    today = datetime.today().strftime("%Y-%m-%d")
    file_name = "item_profile_%s.pkl" % today
    if os.path.exists(file_name):
        return pd.read_pickle(file_name)
    impala_cursor.execute(sql)
    df = as_pandas(impala_cursor)
    df.to_pickle(file_name)
    return df
