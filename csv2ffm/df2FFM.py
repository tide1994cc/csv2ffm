# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:37:20 2019

@author: tide
"""
from sklearn.preprocessing import MinMaxScaler
import time 
import os
import json


class ffm():
    def __init__(self,df,categorCol,labelCol):
        self.df=df  #df
        self.columns=df.columns
        self.categorCol=categorCol #分类变量的列名
        self.labelCol=labelCol   #label 列名
        self.continusCol=self.columns.drop(self.categorCol).drop(self.labelCol)
        self.fieldDict=None   #field 词典
        self.fieldDictVerse=None
        self.onehotDict=None  #onehote 词典
        self.onehotDictVerse=None
        self.length=len(df)
        self.path='ffm.txt'
        self.processing()
    
    # 连续变量需要归一化处理           
    def _ffmNormalization(self):
        mm=MinMaxScaler()
        for col in self.columns.drop(self.categorCol):
            self.df[col]=mm.fit_transform(self.df[col].values.reshape(-1,1))
    
    #离散变量的处理
    def _categorPre(self):
        for col in self.categorCol:
            self.df[col]=self.df[col].apply(lambda x: str(col)+'_'+str(x))
    
     # 生成field 和onehot字典，并且保存
    def _colDict(self):
        field=[]
        field=self.columns.drop(self.labelCol)
        self.fieldDict=dict(zip(field,range(len(field))))
        self.fieldDictVerse=dict(zip(range(len(field)),field))
        
        onehot=[]
        for col in self.categorCol:
            onehot.extend(self.df[col].unique())
        
        onehot.extend(self.continusCol)
        self.onehotDict=dict(zip(onehot,range(len(onehot))))
        self.onehotDictVerse=dict(zip(range(len(onehot)),onehot))

        path='./data/Json'
        isExists=os.path.exists(path)
        if not isExists:
                 os.makedirs(path) 
        
        #保存在json中，方便以后使用
        with open('./data/Json/fieldDict.json','w') as f:
                json.dump(self.fieldDict,f)     
        
        with open('./data/Json/fieldDictVerse.json','w') as f:
                json.dump(self.fieldDictVerse,f)  
        
        with open('./data/Json/onehotDict.json','w') as f:
                json.dump(self.onehotDict,f)  
        
        with open('./data/Json/onehotDictVerse.json','w') as f:
                json.dump(self.onehotDictVerse,f)  
  
    #分类变量处理
    def _categorProcessing(self):
        for col in self.categorCol:   
            fieldName=self.fieldDict[col]
            self.df[col]=self.df[col].apply(lambda x: '{}:{}:1'.format(fieldName, self.onehotDict[x]))
        
    #连续变量梳理
    def _continusProcessing(self):
        for col in self.continusCol:
            fieldName=self.fieldDict[col]
            onehotName=self.onehotDict[col]
            self.df[col]=self.df[col].apply(lambda x:'{}:{}:{}'.format(fieldName, onehotName,x))
   
    def processing(self):
        start=time.time()
        self.path='./data/ffm.txt'
        
        self._ffmNormalization()
        self._categorPre()
        self._colDict()
        
                
        self._categorProcessing()
        self._continusProcessing()

       
        self.df.insert(0,'label',self.df[self.labelCol]) #将label 置于首列
        self.df=self.df.drop(columns=self.labelCol)
        
        self.df.to_csv(self.path,index=False,header=False)
        end=time.time()
        timeCost=end-start
        print("格式转换完成，总耗时为：{}".format(timeCost))
        print("输出文件保存为：{}".format(self.path))


    
        
             

        