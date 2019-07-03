# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:56:06 2019

@author: tide
"""

import pandas as pd 
import df2FFM


movie=pd.read_csv('./data/anime.csv')

movie=movie.dropna()
movie['genre']=movie['genre'].apply(lambda x:x.split(',')[0])
movie=movie.drop(columns=['name','anime_id'])


categor=['type','episodes','genre']
labelCol=['rating']



ffmForm=df2FFM.ffm(movie,categor,labelCol)


ffmForm.fieldDictVerse
ffmForm.onehotDictVerse

