from __future__ import print_function
import pymysql
import random
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


def LSH(spot, recommend_num):
    spark = SparkSession \
        .builder \
        .appName("BucketedRandomProjectionLSHExample") \
        .getOrCreate()

    '''
    # 正则化后的所有景点数据
    data = [('嵛山岛', Vectors.dense([0.2, 0.5, 0.7, 0.5]),),
            ('仙山牧场', Vectors.dense([0.4, 0.4, 0.1, 0.4]),),
            ('大洲岛', Vectors.dense([0.5, 0.1, 0.1, 0.5]),),
            ('御茶园', Vectors.dense([0.2, 0.4, 0.3, 0.6]),),
            ('洞宫山', Vectors.dense([0.3, 0.1, 0.2, 0.2]),),
            ('玉女峰', Vectors.dense([0.4, 0.4, 0.5, 0.4]),),
            ('翡翠谷', Vectors.dense([0.6, 0.1, 0.1, 0.5]),),
            ('白云寺', Vectors.dense([0.9, 0.1, 0.2, 0.1]),),
            ('泰宁地质博物苑', Vectors.dense([0.7, 0.1, 0.3, 0.7]),),
            ('晒布岩', Vectors.dense([1, 0.4, 0.5, 0.4]),)]
    '''

    df = spark.createDataFrame(data, ["name", "features"])
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0, numHashTables=3)
    model = brp.fit(df)

    # key = Vectors.dense([0.5, 0.8, 0.1, 0.5])  # 做推荐的景点
    # model.approxNearestNeighbors(df, key, 3).show()  # 对此景点推荐3个最相似的景点
    result = model.approxNearestNeighbors(df, data.get(spot), recommend_num)

    spark.stop()

    return result


if __name__ == "__main__":

    db = pymysql.connect(host="114.116.43.151", user="root", password="sspku02!", db="qunar_db")
    print("连接成功！")

    cursor = db.cursor()
    cursor.execute('select * from site order by id;')
    site_names = []
    for i in range(18510):
        site_names.append(cursor.fetchone()[1])
    print(site_names[:5])
    print(site_names[-5:])
    print(site_names[9521])
    print(len(site_names))

    '''
    cursor.execute("DROP TABLE IF EXISTS site_recommend")
    create_table = """CREATE TABLE site_recommend(
                site_name VARCHAR(50) NOT NULL,
                recommend_1 VARCHAR(50) NOT NULL,
                recommend_2 VARCHAR(50) NOT NULL,
                recommend_3 VARCHAR(50) NOT NULL,
                recommend_4 VARCHAR(50) NOT NULL,
                recommend_5 VARCHAR(50) NOT NULL,
                recommend_6 VARCHAR(50) NOT NULL,
                recommend_7 VARCHAR(50) NOT NULL,
                recommend_8 VARCHAR(50) NOT NULL,
                recommend_9 VARCHAR(50) NOT NULL,
                recommend_10 VARCHAR(50) NOT NULL,
                recommend_11 VARCHAR(50) NOT NULL,
                recommend_12 VARCHAR(50) NOT NULL,
                recommend_13 VARCHAR(50) NOT NULL,
                recommend_14 VARCHAR(50) NOT NULL,
                recommend_15 VARCHAR(50) NOT NULL,
                recommend_16 VARCHAR(50) NOT NULL,
                recommend_17 VARCHAR(50) NOT NULL,
                recommend_18 VARCHAR(50) NOT NULL,
                recommend_19 VARCHAR(50) NOT NULL,
                recommend_20 VARCHAR(50) NOT NULL
            )"""
    cursor.execute(create_table)
    print("建表成功！")
    '''

    suc_cnt = 0
    i = 0
    while i < 18510:
    # for i in range(18510):
        sql = "INSERT INTO site_recommend (site_name, " \
              "recommend_1, recommend_2, recommend_3, recommend_4, recommend_5, " \
              "recommend_6, recommend_7, recommend_8, recommend_9, recommend_10, " \
              "recommend_11, recommend_12, recommend_13, recommend_14, recommend_15, " \
              "recommend_16, recommend_17, recommend_18, recommend_19, recommend_20) " \
              "VALUE (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "

        recommend = LSH(site_names[i], 20)

        try:
            cursor.execute(sql, ([site_names[i],
                                  recommend[0], recommend[1], recommend[2], recommend[3], recommend[4],
                                  recommend[5], recommend[6], recommend[7], recommend[8], recommend[9],
                                  recommend[10], recommend[11], recommend[12], recommend[13], recommend[14],
                                  recommend[15], recommend[16], recommend[17], recommend[18], recommend[19]]))

            db.commit()
            #if cnt % 100 == 0:
            print(i, "is success!")
            suc_cnt += 1
            i += 1
        except:
            db.rollback()

            print(i, "failed!")
    db.close()
    print(suc_cnt)
