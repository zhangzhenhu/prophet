#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 zhangzhenhu, Inc. All Rights Reserved
#

# from __future__ import print_function
import sys
import json
import os
import argparse
from os.path import expanduser, join
import ahocorasick
import collections
import cPickle
from pyspark.sql import SparkSession
from pyspark.sql import Row, Column
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import datetime

# reload(sys)
# sys.setdefaultencoding('utf-8')

subject_name = {
    1: u'语文',
    2: u'数学',
    3: u'英语',
    4: u'物理',
    5: u'化学',
    6: u'生物',
    8: u'历史',
    9: u'地理',
    10: u'政治',
}
grade_name = {
    1: u'小学',
    2: u'初中',
    3: u'高中',
}
question_type = {
    '1': u"单选",
    '2': u"多选",
    '3': u"多选多",
    '4': u"填空",
    '5': u"判断",
    '6': u"配对",
    '7': u"排序",
    '8': u"解答",
    '9': u"复合",
    '10': u"完形填空",
    '11': u"语音跟读",
    '12': u"口语题",
    '13': u"连词成句",
}


def student_answer(spark):
    """
    统计每道题目的学生作答人次
    :param spark:
    :return:
    """
    ################
    # 学生答题记录，
    #################
    stu_answer = spark.sql(
        '''select 
            sa.sa_stu_id as stu_id,
            sa.sa_qst_id as qst_id,
            sa.sa_answer_status as answer_status,
            sa.sa_answer_cont as answer_cont,
            sa.sa_year as year,
            sa.sa_grd_id as grd_id,
            sa.sa_term_id as term_id,
            sa.sa_subj_id as subj_id,
            sa.sa_lev_id as lev_id,
            sa.sa_questiontypestatus as questiontypestatus, 
            sa.sa_create_time as create_time,
            lq.lq_origin_id as tk_id,
            lq.lq_qst_type,
            lq.lq_qst_difct ,
            tb_qst_kh.kh_ids
        from ods_ips_tb_stu_answer sa
        join ods_ips_tb_level_question lq on  lq.lq_id = sa.sa_lq_id
        join tb_qst_kh on tb_qst_kh.tk_id = lq.lq_origin_id
        where
            lq.lq_library_id==5
        ''')
    # stu_answer.cache()
    stu_answer.createOrReplaceTempView('tb_stu_answer')
    # total_count = stu_answer.count()

    # 输出到hdfs
    # save_path = options.setting['HDFS_DATA_HOME'] + '/' + 'stu_answer'
    # stu_answer.coalesce(20).write.save(path=save_path, compression="gzip", format='json', mode='overwrite')
    # print  "saved stu_answer to %s" % save_path
    #############
    # 题目答题情况
    ##############

    qst_answer = spark.sql('''
        select
            tk_id, 
            sum(if(a.answer_status=0,1,0)) ans_undo,
            sum(if(a.answer_status=1,1,0)) ans_right,
            sum(if(a.answer_status=2,1,0)) ans_error,
            sum(if(a.answer_status=3,1,0)) ans_timeout,
            count(1) as ans_total
        from tb_stu_answer a
        group by tk_id

    ''')
    # qst_answer.cache()
    # print "tb_stu_answer total count:", qst_answer.count()
    return qst_answer


def group_question_kh(spark):
    """
    聚合题目的知识点
    :param spark:
    :return:
    """
    ##
    # 题目知识点
    # 聚合题目知识点，一道题目一行数据，每道题多个知识点逗号连接
    ##
    qst_kh = spark.sql("""
    select 
        qkh.question_id as tk_id,
        concat_ws(',',collect_set(kh_id)) as kh_ids,
        collect_list(kh.id) as kh_id,
        collect_list(kh.name) as kh_name,
        collect_list(kh.parent_id) as kh_parent_id,
        collect_list(kh.root_id) as kh_root_id,
        collect_list(kh.degree) as kh_degree,
        collect_list(kh.grade_id) as kh_grade_id,
        collect_list(kh.sort) as kh_sort ,
        collect_list(kh.new_subject_id) as kh_subject_id,
        collect_list(kh.modify_time) as kh_modify_time
    from ods_tk_question_knowledge_hierarchy qkh
    join ods_tk_knowledge_hierarchy kh on kh.id = qkh.kh_id
    group by qkh.question_id

    """)
    # qst_kh.cache()
    total_count = qst_kh.count()
    print ("tb_qst_kh total count:", total_count)
    return qst_kh
    # 输出到hdfs
    # save_path = options.setting['HDFS_DATA_HOME'] + '/' + 'question_kh'
    # qst_kh.coalesce(5).write.save(path=save_path, compression="gzip", format='json', mode='overwrite')
    # print  "saved tb_qst_kh to %s" % save_path


def mk_kh_ac_and_tree(spark):
    """
    创建知识点的树形结构和ac自动机
    :param spark:
    :return:
    """
    print ("mk_kh_ac_and_tree..")
    df_kh = spark.sql("""
        select 
            id,
            name,
            grade_id,
            new_subject_id as subject_id ,
            parent_id,
            degree,
            root_id
        from odata.ods_tk_knowledge_hierarchy
    """)

    kh_list = df_kh.collect()
    global g_automatons, g_kh_tree, g_kh_dict
    # 他妹的ahocorasick这个库Python2.7的版本不支持unicode，string是无法支持多字节的中文匹配的
    # 暂时先改成包里搜索吧，好在题目题干不是很长
    # g_automatons = {"%s_%s" % (i, j): ahocorasick.Automaton() for i in subject_name for j in grade_name}
    # g_kh_tree = collections.defaultdict(list)
    g_kh_dict = collections.defaultdict(dict)
    # g_kh_list = collections.defaultdict(list)

    # 针对每个（学科，学部）的知识点建立AC自动机，和树形结构
    for record in kh_list:
        key = "%s_%s" % (record.subject_id, record.grade_id)
        # a = g_automatons[key]
        # a.add_word(record['name'].encode('UTF-8'), record.asDict())  # 这里必须copy一份
        # g_kh_tree[key].append(record.asDict())  # 这里必须copy一份
        g_kh_dict[key][record['id']] = record.asDict()  # 这里必须copy一份

    # for key, records in g_kh_tree.iteritems():
    # g_kh_list = records
    # 建立知识点树
    # g_kh_tree[key] = build_tree(records)
    # 生成状态机
    # g_automatons[key].make_automaton()

    # 保存下来
    save_path = options.setting.get("DATA_PATH")
    # with open(os.path.join(save_path, 'knowledge_ac.pickle'), 'w') as fh:
    #
    #     cPickle.dump(g_automatons, fh)
    with open(os.path.join(save_path, 'knowledge_dict.pickle'), 'w') as fh:
        cPickle.dump(g_kh_dict, fh)
    print ("mk_kh_ac_and_tree done.")


def main(options):
    spark = SparkSession \
        .builder \
        .appName("%s.%s.%s" % ("prophet","irt" , "zhangzhenhu")) \
        .config("spark.executor.instances", 100) \
        .config("spark.executor.cores", 1) \
        .config("spark.executor.memory", '3G') \
        .config("spark.driver.memory", '4G') \
        .config("spark.sql.hive.convertMetastoreParquet", 'false') \
        .enableHiveSupport() \
        .getOrCreate()
    # .config("spark.yarn.dist.archives",
    #         'hdfs://iz2ze7u02k402bnxra1j1xz:8020/user/app_bi/tools/python27.jar#python27') \
    # .config("spark.pyspark.python", './python27/bin/python2') \

    # 设置默认数据库odata
    spark.catalog.setCurrentDatabase('odata')

    # step 1 创建知识点的ac自动机和树形结构
    mk_kh_ac_and_tree(spark)

    # step 2 聚合题目的知识点
    group_question_kh(spark).createOrReplaceTempView('tb_qst_kh')

    # step 3 统计每道题目的学生作答人次
    student_answer(spark).createOrReplaceTempView('tb_qst_answer')

    ################
    # 综合题目画像
    ###############
    for xueke in ['sx',  # 数学
                  'yw',  # 语文
                  'yy',  # 英语
                  'zz',  # 政治
                  'wl',  # 物理
                  'hx',  # 化学
                  'sw',  # 生物
                  'ls',  # 历史
                  'dl',  # 地理
                  ]:
        spark.sql("""
        select 
            que_id,
            collect_list(ao_id) as ao_id,
            collect_list(content) as ao_content,
            collect_list(modity_date) as ao_modity_date,
            collect_list(sort) as ao_sort,
            collect_list(ao_val) as ao_val
        
        from ods_tk_tk_answer_option_%s tb
        where state==0
        group by tb.que_id
        
        """ % xueke).createOrReplaceTempView('tb_ao')

        qst_profile = spark.sql('''
            select
                xk.*,
                tb_qst_answer.ans_undo,
                tb_qst_answer.ans_right,
                tb_qst_answer.ans_error,
                tb_qst_answer.ans_timeout,
                tb_qst_answer.ans_total,
                tb_qst_kh.kh_ids,
                tb_qst_kh.kh_id,
                tb_qst_kh.kh_name,
                tb_qst_kh.kh_parent_id,
                tb_qst_kh.kh_root_id,
                tb_qst_kh.kh_degree,
                tb_qst_kh.kh_grade_id,
                tb_qst_kh.kh_sort,
                tb_qst_kh.kh_subject_id,
                tb_qst_kh.kh_modify_time,
                tb_ao.ao_id,
                tb_ao.ao_content,
                tb_ao.ao_modity_date,
                tb_ao.ao_sort,
                tb_ao.ao_val,
                '%s' as subject_name
                
            from ods_tk_tk_question_%s xk
            left join tb_qst_answer on tb_qst_answer.tk_id == xk.que_id
            left join tb_qst_kh on tb_qst_kh.tk_id == xk.que_id
            left join tb_ao on tb_ao.que_id == xk.que_id
            where xk.state = 0 
        
        
        ''' % (xueke, xueke))
        # 给题目扩展知识点标签
        # qst_profile = qst_profile.withColumn('kh_trees',
        #                                      patch_kh_udf(
        #                                          qst_profile['subject_id'],
        #                                          qst_profile['grade_group_id'],
        #                                          qst_profile['qt_id'],
        #                                          qst_profile['content'],
        #
        #                                          qst_profile['kh_id'],
        #                                          qst_profile['kh_name'],
        #                                          qst_profile['kh_parent_id'],
        #                                          qst_profile['kh_root_id'],
        #                                          qst_profile['kh_degree'],
        #                                          qst_profile['kh_sort'],
        #                                      ))
        # qst_profile.cache()
        # total_count = qst_profile.count()
        # 建树逻辑和上面的放在一起会导致spark节点内存超限，原因未找到。
        # qst_profile = qst_profile.withColumn('kh_trees', nimeide_udf(qst_profile['kh_trees']))
        # target_tb = spark.table('dmdb.dm_qstportrait_qst')
        # final_data = qst_profile.select(*target_tb.columns)
        # 每次更新完，需要手动刷新一下impala的缓存索引才行
        # final_data.write.insertInto('dmdb.dm_qstportrait_qst', overwrite=True)

        final_data = qst_profile.fillna({'ans_undo': 0,
                                         'ans_right': 0,
                                         'ans_error': 0,
                                         'ans_total': 0,
                                         'ans_timeout': 0,
                                         # 'ao_id': None,
                                         # 'ao_content': None,
                                         # 'ao_modity_date': None,
                                         # 'ao_sort': None,
                                         # 'ao_val': None,
                                         # 'kh_ids': None,
                                         # 'kh_id': None,
                                         # 'kh_name': None,
                                         # 'kh_parent_id': None,
                                         # 'kh_root_id': None,
                                         # 'kh_degree': None,
                                         # 'kh_grade_id': None,
                                         # 'kh_sort': None,
                                         # 'kh_subject_id': None,
                                         # 'kh_modify_time': None,
                                         })
        # 输出到hdfs，其它任务使用这份数据
        save_path = options.setting['HDFS_STRATEGY_HOME'] + '/' + 'question_portrait/basic/' + xueke
        final_data.write.save(path=save_path, compression="gzip", format='json', mode='overwrite')
        print( "saved %s to hdfs %s" % (xueke, save_path))
        # qst_profile
        # print  "saved stu_answer to %s" % save_path
    spark.stop()


def init_option():
    """
    初始化命令行参数项
    Returns:
        OptionParser 的parser对象
    """

    # OptionParser 自己的print_help()会导致乱码，这里禁用自带的help参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action="store_true", default=False,
                        dest="debug",
                        help=u"开启debug模式")
    parser.add_argument("-i", "--input", dest="input",
                        help=u"输入路径")
    parser.add_argument("-o", "--output", dest="output",
                        help=u"hive table name")
    return parser


if __name__ == '__main__':
    sys.path.append('.')
    sys.path.append('..')

    parser = init_option()
    options = parser.parse_args()
    # if options.help:
    #     # OptionParser 自己的print_help()会导致乱码
    #     usage = parser.format_help()
    #     print(usage.encode("UTF-8"))
    #     quit()
    # if options.input:
    #
    #     options.input = open(options.input)
    # else:
    #     options.input = sys.stdin

    main(options)
