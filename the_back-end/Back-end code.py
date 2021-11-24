from flask import request,Flask,jsonify,render_template,Response,send_file,make_response
from werkzeug.utils import secure_filename
import os
import shutil
import asyncio
from os.path import join,getsize
#coding:utf8
import re
import cv2
import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
import pickle, sys
import scipy.misc as im
import imageio
import matplotlib.image as mp
import urllib.request
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from PIL import Image
from skimage.transform import resize
K.set_image_data_format('channels_last')
import zipfile
from flask_cors import *
from flask_cors import CORS
import pymysql
import datetime
import json



app=Flask(__name__)
CORS(app, supports_credentials=True)
#CORS(app, resources=r'/*')
ALLOWED_EXTENSIONS=set(['png','jpg','bmp'])



def data_augmentation(type, file_path, file_name,params):
    input_file=file_path+'/'+file_name
    MUTATION_RATE=0.5
    GAUSS_RATE=0.5
    params=int(params)
    global X_augmentation
    X_augmentation=[]
    global Y_augmentation
    Y_augmentation=[]

    with open(input_file,'r') as load_f:
        be=0
        while True:
            line=load_f.readline()
            if not line:
                break
            load_dict = json.loads(line)
            X_augmentation.append(load_dict["data"])
            Y_augmentation.append(load_dict["label"])
            be=be+1

    X_augmentation=np.array(X_augmentation)
    Y_augmentation=np.array(Y_augmentation)
    for i in range(X_augmentation.shape[0]):
        print(X_augmentation[i])
        print(Y_augmentation[i])
    if type==5:#Zooming
        params=params*0.005
        params=params+1
        if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            for i in range(X_augmentation.shape[0]):
                for j in range(X_augmentation.shape[2]):
                    if np.random.rand() < GAUSS_RATE:
                        X_augmentation[i][0][j]=X_augmentation[i][0][j]*params
                        X_augmentation[i][1][j]=X_augmentation[i][1][j]*params
                    else:
                        X_augmentation[i][0][j]=X_augmentation[i][0][j]*params
                        X_augmentation[i][1][j]=X_augmentation[i][1][j]*params
    elif type==4:#Sequence Transformation
        if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            allnum=6*params
            for i in range(allnum):
                a=random.randint(0,127)
                b=random.randint(0,127)
                for ii in range(X_augmentation.shape[0]):
                    if np.random.rand() < GAUSS_RATE:
                        uu=X_augmentation[ii][0][a]
                        X_augmentation[ii][0][a]=X_augmentation[ii][0][b]
                        X_augmentation[ii][0][b]=uu
                        vv=X_augmentation[ii][1][a]
                        X_augmentation[ii][1][a]=X_augmentation[ii][1][b]
                        X_augmentation[ii][1][b]=vv
    elif type==1:#Gaussian Noise
        global sum1
        global sum2
        global all_num
        global cal_sum1
        global cal_sum2
        global aa
        global bb
        global AA
        global BB
        sum1=0
        sum2=0
        all_num=0
        cal_sum1=0
        cal_sum2=0
        for i in range(X_augmentation.shape[0]):
            for j in range(X_augmentation.shape[2]):
                sum1=sum1+X_augmentation[i][0][j]
                sum2=sum2+X_augmentation[i][1][j]
                all_num=all_num+1
        sum1=sum1/all_num
        sum2=sum2/all_num
        for i in range(X_augmentation.shape[0]):
            for j in range(X_augmentation.shape[2]):
                cal_sum1=cal_sum1+(sum1-X_augmentation[i][0][j])*(sum1-X_augmentation[i][0][j])
                cal_sum2=cal_sum2+(sum2-X_augmentation[i][1][j])*(sum2-X_augmentation[i][1][j])
        cal_sum1=cal_sum1/all_num
        cal_sum2=cal_sum2/all_num

        AA=cal_sum1
        BB=cal_sum2
        aa=sum1
        bb=sum2
        print('aa',aa)
        print('bb',bb)
        print('cc',AA)
        print('dd',BB)
        print('params',params)
        for i in range(X_augmentation.shape[0]):
            for j in range(X_augmentation.shape[2]):
                use=np.random.rand()
                if np.random.rand() < GAUSS_RATE:
#                    print('X_augmentation[i][0][j]_be',X_augmentation[i][0][j])
                    X_augmentation[i][0][j]=X_augmentation[i][0][j]+random.gauss(aa,AA)*params
                    X_augmentation[i][1][j]=X_augmentation[i][1][j]+random.gauss(bb,BB)*params
#                    print('X_augmentation[i][0][j]',X_augmentation[i][0][j])
    elif type==2:#Random Erase
        allnum=6*params
        use=X_augmentation.shape[0]
        for i in range(allnum):
            a=random.randint(0,use-1)
            b=random.randint(0,use-1)
            aa=random.randint(0,127)
            bb=random.randint(0,127)
            if np.random.rand() < GAUSS_RATE:
                X_augmentation[a][0][aa]=0
                X_augmentation[a][1][aa]=0
                X_augmentation[b][0][bb]=0
                X_augmentation[b][1][bb]=0
    elif type==3:#Fuzzy
        allnum=6*params
        use=X_augmentation.shape[0]
        for i in range(allnum):
            a=random.randint(0,use-1)
            b=random.randint(0,use-1)
            for w in range(4):
                j=random.randint(1,125)
                X_augmentation[a][0][j+1]=(X_augmentation[a][0][j]+X_augmentation[a][0][j+1]+X_augmentation[a][0][j+2])/3.0
                X_augmentation[a][1][j+1]=(X_augmentation[a][1][j]+X_augmentation[a][1][j+1]+X_augmentation[a][1][j+2])/3.0
                X_augmentation[b][0][j+1]=(X_augmentation[b][0][j]+X_augmentation[b][0][j+1]+X_augmentation[b][0][j+2])/3.0
                X_augmentation[b][1][j+1]=(X_augmentation[b][1][j]+X_augmentation[b][1][j+1]+X_augmentation[b][1][j+2])/3.0
    else:
        print("there is no this type.")

    output_file=file_path+'/'+str(type)+"out_"+file_name
    print(X_augmentation.shape)
    print(Y_augmentation.shape)
    #info=[[[123,22,'bb'],[234,'cc','dd']],[[456,'aa','bb'],[234,'cc','dd']]]
    Result=[]
    all={}
    tops={}
#    with open(output_file, 'w') as files:
#        files.seek(0)
#        files.truncate()
#        w=0
#        for i in range(X_augmentation.shape[0]):
#            w=w+1
#            all['data']=X_augmentation[i].tolist()
#            all['label']=Y_augmentation[i].tolist()
#            files.write(json.dumps(all)+'\n')
#        files.write(json.dumps(tops)+'\n')
#    with open(output_file, 'w') as files:
#        for i in range(X_augmentation.shape[0]):
#            all['data']=X_augmentation[i].tolist()
#            all['label']=Y_augmentation[i].tolist()
#    #        all['snr']=(np.array(str(snr))).tolist()
#            files.write(json.dumps(all)+'\n')
    with open(output_file, 'w') as files:
        files.seek(0)
        files.truncate()
        w=0
        for i in range(X_augmentation.shape[0]):
            w=w+1
            all['data']=X_augmentation[i].tolist()
            all['label']=Y_augmentation[i].tolist()
            files.write(json.dumps(all)+'\n')
#        all['snr']=(np.array(str(snr))).tolist()
            


@app.route('/')
def hello():
    return 'hello world'
    

@app.route('/uploadwave',methods=['GET','POST'])
def upload_wave_file():
    db=pymysql.connect(host='129.211.89.155',port=33000,user='root',password='12345678',db='hjw')
    cur=db.cursor()
    file_path='/home/ise/dl/imageandwave/upload_file'
    print(123)
    print(request.headers)
    if request.method=='POST':
#        result_text = {"statusCode": 200,"message": "文件上传成功"}
#        response = make_response(jsonify(result_text))
#        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin') or 'http://127.0.0.1:5000'
#        response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,HEAD,GET,POST'
#        response.headers['Access-Control-Allow-Headers'] = 'x-requested-with'
        #读取前段发来的数据
#        json_data = json.loads(request.get_data().decode('utf-8'))
#        augmentation_type_name=json_data.get('augmentation_type_name')
#        augmentation_method_name=json_data.get('augmentation_method_name')
        message={
            "page":200,
            "original":[],
            "detailsone":[],
            "detailstwo":[],
            "detailsthree":[],
            "detailsfour":[],
            "detailsfive":[]
        }
        augment_type_one=request.form.get('augment_type_one')
        augment_type_two=request.form.get('augment_type_two')
        augment_type_three=request.form.get('augment_type_three')
        augment_type_four=request.form.get('augment_type_four')
        augment_type_five=request.form.get('augment_type_five')
        params=request.form.get('params')
        params=str(params)
        files=request.files['file']
        files_name=files.filename
        print(augment_type_one)
        print(augment_type_two)
        print(augment_type_three)
        print(augment_type_four)
        print(augment_type_five)
        print(params)
        #给任务编号id
        sql="select max(id) from Task"
        cur.execute(sql)
        result=cur.fetchall()
        for (item,) in result:
            id=item+1
        print(id)
        #params已经存在，即前段读取的params

        #status初始化   0：未处理   1：正在处理    2:已完成    3:异常
        status=0
        #给数据存储位置origin_file_path
        files_name=str(id)+files_name
        origin_file_path=file_path

        #name为文件名
        name=files_name

        #初始化任务状态
        is_deleted=0

        #获取augmentation_method_id
#        augmentation_method_id=1
        #默认master_id=123
        master_id=123

        #初始化上传时间
        update_datetime=datetime.datetime.now()


        #保存文件
        save_path=origin_file_path+'/'+name
        files.save(save_path)
        print(save_path)
        print(id)
        print(params)
        print(status)
        print(origin_file_path)
        print(is_deleted)
        print(master_id)
        print(update_datetime)
        print(name)
        print(augment_type_one)
        print(augment_type_two)
        print(augment_type_three)
        print(augment_type_four)
        print(augment_type_five)
        print(save_path)
        print(str(id))
        print(str(params))
        print(str(status))
        print(str(origin_file_path))
        print(str(is_deleted))
        print(str(master_id))
        print(str(update_datetime))
        print(str(name))
        print(str(augment_type_one))
        print(str(augment_type_two))
        print(str(augment_type_three))
        print(str(augment_type_four))
        print(str(augment_type_five))
        sql="insert into Task(id,params,status,origin_file_path,is_deleted,master_id,update_datetime,name,augment_type_one,augment_type_two,augment_type_three,augment_type_four,augment_type_five) values("+str(id)+","+str(params)+","+str(status)+",'"+str(origin_file_path)+"',"+str(is_deleted)+","+str(master_id)+",'"+str(update_datetime)+"','"+str(name)+"',"+str(augment_type_one)+","+str(augment_type_two)+","+str(augment_type_three)+","+str(augment_type_four)+","+str(augment_type_five)+")"
        cur.execute(sql)
        db.commit()


        # cur.close()
        # db=pymysql.connect(host='129.211.89.155',port=33000,user='root',password='12345678',db='hjw')
        # cur=db.cursor()
        #处理所有的任务
        sql="select * from Task where status = 0"
        cur.execute(sql)
        result=cur.fetchall()
        for item in result:
            top_id=item[0]
            top_params=item[1]
            top_status=item[2]
            top_origin_file_path=item[3]
            top_fin_datetime=item[8]
            top_is_deleted=item[4]
            top_master_id=item[5]
            top_start_datetime=item[6]
            top_update_datetime=item[7]
            top_name=item[9]
            method_one=item[10]
            method_two=item[11]
            method_three=item[12]
            method_four=item[13]
            method_five=item[14]
            #修改top任务的状态以及开始的时间
            top_status=1
            top_start_datetime
            sql="update Task set status="+str(top_status)+", start_datetime='"+str(top_start_datetime)+"' where id = "+str(top_id)
            cur.execute(sql)
            db.commit()

            #运行Task
            generate_file1_path=file_path+'/'+"1out_"+name
            generate_file2_path=file_path+'/'+"2out_"+name
            generate_file3_path=file_path+'/'+"3out_"+name
            generate_file4_path=file_path+'/'+"4out_"+name
            generate_file5_path=file_path+'/'+"5out_"+name
            if method_one==1:
                type=1
                data_augmentation(type,top_origin_file_path,top_name,params)
                sql="update Task set generate_file1_path='"+generate_file1_path+"' where id = "+str(top_id)
                cur.execute(sql)
                db.commit()
            if method_two==1:
                type=2
                data_augmentation(type,top_origin_file_path,top_name,params)
                sql="update Task set generate_file2_path='"+generate_file2_path+"' where id = "+str(top_id)
                cur.execute(sql)
                db.commit()
            if method_three==1:
                type=3
                data_augmentation(type,top_origin_file_path,top_name,params)
                sql="update Task set generate_file3_path='"+generate_file3_path+"' where id = "+str(top_id)
                cur.execute(sql)
                db.commit()
            if method_four==1:
                type=4
                data_augmentation(type,top_origin_file_path,top_name,params)
                sql="update Task set generate_file4_path='"+generate_file4_path+"' where id = "+str(top_id)
                cur.execute(sql)
                db.commit()
            if method_five==1:
                type=5
                data_augmentation(type,top_origin_file_path,top_name,params)
                sql="update Task set generate_file5_path='"+generate_file5_path+"' where id = "+str(top_id)
                cur.execute(sql)
                db.commit()

            print("finished")
            #设置终止时间
            top_fin_datetime=datetime.datetime.now()
            #设置任务状态为完成
            top_status=2
            #提交执行命令
            sql="update Task set status="+str(top_status)+", fin_datetime='"+str(top_fin_datetime)+"' where id = "+str(top_id)
            cur.execute(sql)
            db.commit()




        # message={
        #     "page":10,
        #     "details":[]
        # }
        # XX_augmentation=[]
        # YY_augmentation=[]
        # with open(generate_file_path,'r') as load_f:
        #     be=0
        #     while True:
        #         line=load_f.readline()
        #         if not line:
        #             break
        #         load_dict = json.loads(line)
        #         dict={
        #             "data":load_dict["data"],
        #             "label":load_dict["label"]
        #         }
        #         message['details'].append(dict)
        #         XX_augmentation.append(load_dict["data"])
        #         YY_augmentation.append(load_dict["label"])
        #         be=be+1

        # XX_augmentation=np.array(XX_augmentation)
        # YY_augmentation=np.array(YY_augmentation)
        # print(XX_augmentation.shape[0])
        return message
    else:
        return "upload get"


@app.route('/downloadwave',methods=['GET','POST'])
def download_wave_file():
    db=pymysql.connect(host='129.211.89.155',port=33000,user='root',password='12345678',db='hjw')
    cur=db.cursor()
    global task_id
    task_id=0
    global fileroot
    fileroot="123"
    global fileroot1
    fileroot1="123"
    global fileroot2
    fileroot2="123"
    global fileroot3
    fileroot3="123"
    global fileroot4
    fileroot4="123"
    global fileroot5
    fileroot5="123"
    if request.method=='GET':
        augment_type_one=request.args.get('augment_type_one')
        augment_type_two=request.args.get('augment_type_two')
        augment_type_three=request.args.get('augment_type_three')
        augment_type_four=request.args.get('augment_type_four')
        augment_type_five=request.args.get('augment_type_five')
        print(str(augment_type_one))
        print(str(augment_type_two))
        print(str(augment_type_three))
        print(str(augment_type_four))
        print(str(augment_type_five))
        if str(augment_type_one)=='1':
            download_method=1
            sql="select max(id) from Task where augment_type_one=1"
            print(sql)
            cur.execute(sql)
            result=cur.fetchall()
            for (item,) in result:
                task_id=item
        if str(augment_type_two)=='1':
            download_method=2
            sql="select max(id) from Task where augment_type_two=1"
            print(sql)
            cur.execute(sql)
            result=cur.fetchall()
            for (item,) in result:
                task_id=item
        if str(augment_type_three)=='1':
            download_method=3
            sql="select max(id) from Task where augment_type_three=1"
            print(sql)
            cur.execute(sql)
            result=cur.fetchall()
            for (item,) in result:
                task_id=item
        if str(augment_type_four)=='1':
            download_method=4
            sql="select max(id) from Task where augment_type_four=1"
            print(sql)
            cur.execute(sql)
            result=cur.fetchall()
            for (item,) in result:
                task_id=item
            print(task_id)
        if str(augment_type_five)=='1':
            download_method=5
            sql="select max(id) from Task where augment_type_five=1"
            print(sql)
            cur.execute(sql)
            result=cur.fetchall()
            for (item,) in result:
                task_id=item
        print(task_id)
        if str(task_id)=='0':
            sql="select * from Task where id="+str(task_id)
            cur.execute(sql)
            result=cur.fetchall()
            for item in result:
                origin_file_path=item[3]
                name=item[9]
                fileroot1=item[15]
                fileroot2=item[16]
                fileroot3=item[17]
                fileroot4=item[18]
                fileroot5=item[19]
        else:
            sql="select * from Task where id="+str(task_id)
            cur.execute(sql)
            result=cur.fetchall()
            for item in result:
                origin_file_path=item[3]
                name=item[9]
                fileroot1=item[15]
                fileroot2=item[16]
                fileroot3=item[17]
                fileroot4=item[18]
                fileroot5=item[19]
        X_augmentation=[]
        Y_augmentation=[]
        if str(augment_type_one)=='1':
            fileroot=fileroot1
        if str(augment_type_two)=='1':
            fileroot=fileroot2
        if str(augment_type_three)=='1':
            fileroot=fileroot3
        if str(augment_type_four)=='1':
            fileroot=fileroot4
        if str(augment_type_five)=='1':
            fileroot=fileroot5
        with open(fileroot,'r') as load_f:
            be=0
            while True:
                line=load_f.readline()
                if not line:
                    break
                load_dict = json.loads(line)
                X_augmentation.append(load_dict["data"])
                Y_augmentation.append(load_dict["label"])
                be=be+1
        X_augmentation=np.array(X_augmentation)
        Y_augmentation=np.array(Y_augmentation)
        print(X_augmentation.shape)
        print(Y_augmentation.shape)
        output_file=origin_file_path+'/'+"download_"+name
        print('fileroot',fileroot)
        print('fileroot1',fileroot1)
        print('fileroot2',fileroot2)
        print('fileroot3',fileroot3)
        print('fileroot4',fileroot4)
        print('fileroot5',fileroot5)
        print('fileroot',output_file)
        #info=[[[123,22,'bb'],[234,'cc','dd']],[[456,'aa','bb'],[234,'cc','dd']]]
        Result=[]
        all={}
        tops={}
        with open(output_file, 'w') as files:
            files.seek(0)
            files.truncate()
            w=0
            for i in range(X_augmentation.shape[0]):
                w=w+1
                all['data']=X_augmentation[i].tolist()
                all['label']=Y_augmentation[i].tolist()
#                print(all['data'])
#                tops[str(w)]=all
#                print(tops)
                files.write(json.dumps(all)+'\n')


#        with open(output_file, 'w') as files:
#        for i in range(X_augmentation.shape[0]):
#            files.seek(0)
#            files.truncate()
#            all['data']=X_augmentation[i].tolist()
#            all['label']=Y_augmentation[i].tolist()
#    #        all['snr']=(np.array(str(snr))).tolist()
#            files.write(json.dumps(all)+'\n')
        
#        with open(output_file, 'w') as files:
#            files.seek(0)
#            files.truncate()
#            w=0
#            for i in range(X_augmentation.shape[0]):
#                w=w+1
#                all['data']=X_augmentation[i].tolist()
#                all['label']=Y_augmentation[i].tolist()
#                files.write(json.dumps(all)+'\n')
#            files.write(json.dumps(tops)+'\n')
        
        files=open(output_file,"rb")
#        response.setHeader("Access-Control-Expose-Headers","Content-Disposition")
        return send_file(files,attachment_filename=output_file,as_attachment=True)
    else:
        return 'There is no such file.'
            
@app.route('/showwavemessage')
def show_wave_message():
    db=pymysql.connect(host='129.211.89.155',port=33000,user='root',password='12345678',db='hjw')
    cur=db.cursor()
    augment_type_one=request.args.get('augment_type_one')
    augment_type_two=request.args.get('augment_type_two')
    augment_type_three=request.args.get('augment_type_three')
    augment_type_four=request.args.get('augment_type_four')
    augment_type_five=request.args.get('augment_type_five')
    global download_method
    download_method=0
    global ss
    ss="123"
    if str(augment_type_one)=='1':
        download_method=1
        ss="one"
    if str(augment_type_two)=='1':
        download_method=2
        ss="two"
    if str(augment_type_three)=='1':
        download_method=3
        ss="three"
    if str(augment_type_four)=='1':
        download_method=4
        ss="four"
    if str(augment_type_five)=='1':
        download_method=5
        ss="five"
    sql="select max(id) from Task where augment_type_"+str(ss)+"=1"
    cur.execute(sql)
    result=cur.fetchall()
    for (item,) in result:
        task_id=item
    message={
        "page":10,
        "original":[],
        "detailsone":[],
        "detailstwo":[],
        "detailsthree":[],
        "detailsfour":[],
        "detailsfive":[]
    }
    global original_file_path
    original_file_path="123"
    global original_name
    original_name="123"
    sql="select * from Task where id = "+str(task_id)
    cur.execute(sql)
    result=cur.fetchall()
    for item in result:
        original_file_path=item[3]
        original_name=item[9]
    upload_file_paths=original_file_path+'/'+original_name
    print("upload_file_paths",upload_file_paths)
    with open(upload_file_paths,'r') as load_f:
        while True:
            line=load_f.readline()
            if not line:
                break
            load_dict = json.loads(line)
            dict={
                "data":load_dict["data"],
                "label":load_dict["label"]
            }
            message['original'].append(dict)
    
    if download_method==1:
        sql="select generate_file1_path from Task where id="+str(task_id)
        cur.execute(sql)
        result=cur.fetchall()
        for (item,) in result:
            file_path=item
        print("file_path",file_path)
        with open(file_path,'r') as load_f:
            while True:
                line=load_f.readline()
                if not line:
                    break
                load_dict = json.loads(line)
                dict={
                    "data":load_dict["data"],
                    "label":load_dict["label"]
                }
                message['detailsone'].append(dict)
    if download_method==2:
        sql="select generate_file2_path from Task where id="+str(task_id)
        cur.execute(sql)
        result=cur.fetchall()
        for (item,) in result:
            file_path=item
        with open(file_path,'r') as load_f:
            while True:
                line=load_f.readline()
                if not line:
                    break
                load_dict = json.loads(line)
                dict={
                    "data":load_dict["data"],
                    "label":load_dict["label"]
                }
                message['detailstwo'].append(dict)
    if download_method==3:
        sql="select generate_file3_path from Task where id="+str(task_id)
        cur.execute(sql)
        result=cur.fetchall()
        for (item,) in result:
            file_path=item
        with open(file_path,'r') as load_f:
            while True:
                line=load_f.readline()
                if not line:
                    break
                load_dict = json.loads(line)
                dict={
                    "data":load_dict["data"],
                    "label":load_dict["label"]
                }
                message['detailsthree'].append(dict)
    if download_method==4:
        sql="select generate_file4_path from Task where id="+str(task_id)
        cur.execute(sql)
        result=cur.fetchall()
        for (item,) in result:
            file_path=item
        with open(file_path,'r') as load_f:
            while True:
                line=load_f.readline()
                if not line:
                    break
                load_dict = json.loads(line)
                dict={
                    "data":load_dict["data"],
                    "label":load_dict["label"]
                }
                message['detailsfour'].append(dict)
    if download_method==5:
        sql="select generate_file5_path from Task where id="+str(task_id)
        cur.execute(sql)
        result=cur.fetchall()
        for (item,) in result:
            file_path=item
        with open(file_path,'r') as load_f:
            while True:
                line=load_f.readline()
                if not line:
                    break
                load_dict = json.loads(line)
                dict={
                    "data":load_dict["data"],
                    "label":load_dict["label"]
                }
                message['detailsfive'].append(dict)
    return message

    
@app.route('/userlogin',methods=['GET','POST'])
def userlogin():
    db=pymysql.connect(host='129.211.89.155',port=33000,user='root',password='12345678',db='hjw')
    cur=db.cursor()
    if request.method=='POST':
        username=request.args.get('username')
        password=request.args.get('password')


        sql="select count(*) from User where username='"+str(username)+"' and password='"+str(password)+"' and is_deleted=0"
        cur.execute(sql)
        idnumber=cur.fetchall()
        for (item,) in idnumber:
            xyz=item
        
        sql="select * from User where username='"+str(username)+"' and password='"+str(password)+"' and is_deleted=0"
        cur.execute(sql)
        result=cur.fetchall()
        for item in result:
            id=item[0]
            nickname=item[2]
            user_group=item[4]
            join_datetime=item[5]
            last_login_datetime=item[6]
        join_datetime=datetime.datetime.now()
        
        if xyz!=0:
            sql="update User set join_datetime='"+str(join_datetime)+"'"
            cur.execute(sql)
            db.commit()
            dict={
                "id":id,
                "nickname":nickname,
                "user_group":user_group,
                "last_login_datetime":last_login_datetime,
                "message":"Login in Success!",
                "yemiantiaozhuan":1
            }
            return dict
            #进行页面跳转
        else:
            dict={
                "message":"Failed Login in! Invalid Login Message!",
                "yemiantiaozhuan":0
            }
            return dict
    else:
        return 'error'




if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8050,debug=True,threaded=False)
        




