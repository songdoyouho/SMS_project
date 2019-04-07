# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 07:31:43 2018

@author: songdoyouho

Dcard spyder 
"""

import os
import requests
import urllib.request
import json
import time

# 整理網址
all_folder = '/home/yopipi-ubuntu/桌面/yes_or_no/dcard'
Dcard_api = 'https://www.dcard.tw/_api/'
Dcard_form = 'forums/'
Dcard_posts = 'posts/'
Dcard_links = 'links'
Dcard_comments = 'comments'
ori_Dcard_folder = '/media/yopipi-ubuntu/SSD/Dcard'
today =time.strftime("%Y_%m_%d", time.localtime())
Dcard_folder = os.path.join(ori_Dcard_folder, today)

Dcard_url = Dcard_api + Dcard_form + 'sex' + '/' + Dcard_posts # 加入要爬取的板

# 先抓前30個文章
resp = requests.get(Dcard_url, params = {'popular': 'false'})  # 用 'before': 'id'來找更之前的文章
resp_j = resp.json()

try:
    os.makedirs(ori_Dcard_folder)
    print('Dcard spyder start !')
except: 
    print('Dcard spyder start !')
    
print(ori_Dcard_folder + '/' + 'data.json')
# 第一次寫檔，其他則讀檔    
if os.path.exists(ori_Dcard_folder + '/' + 'data.json'):
    with open(ori_Dcard_folder + '/' + 'data.json', 'r') as f:
        data = json.load(f)
        print('read')
else:
    data = resp_j[29]['id']
    with open(ori_Dcard_folder + '/' + 'data.json', 'w') as f:
        json.dump(data, f)
        print('write')


flag = True
while flag:
    if resp_j[len(resp_j) - 1]['id'] > data:
        print(resp_j[len(resp_j) - 1]['id'])
        resp = requests.get(Dcard_url, params = {'before': str(resp_j[len(resp_j) - 1]['id']),'popular': 'false'})
        resp_j += resp.json()
    else:
        flag = False
        

for tmp in resp_j:
    tmp_id = tmp['id']
    
    if tmp_id > data:
        print(tmp_id, data)
        Dcard_story = Dcard_api + Dcard_posts + str(tmp_id)  # 拿文章內容
        #Dcard_story = Dcard_api + Dcard_posts + str(228352995)  # 拿文章內容
        story = requests.get(Dcard_story)
        story_j = story.json()
        story_folder = Dcard_folder + '/' + story_j['title'] 
        
        # 拿圖片連結
        test_media = story_j['media']
        
        if test_media:
            test_url = story_j['media'][0]['url']
        
            if test_url:    
                try:
                    print(story_folder)
                    os.makedirs(story_folder)
                except Exception as e:
                    print(e)
        
                try:
                    links_j = story_j['media'] # 拿文章裡面的連結們
                except IndexError: # 如果沒有連結
                    print('There is no link in this story!')
                else: # 如果有連結
                    for url in links_j:
                        link = url['url']
                        print('post',link)
                        aaa = link.split('//')[1]
                        bbb = aaa.split('/')[1] # get output file name! 
                        try:
                            urllib.request.urlretrieve(link, story_folder + '/' + bbb) # 儲存圖片 (url, route)
                            urllib.request.urlretrieve(link, all_folder + '/' + bbb) # 儲存圖片 (url, route)
                        except Exception as e:
                            print(e)
                            
        # 找回文中的圖片連結
        params = {}   
        comments_j = []                         
        while True:
            comments = requests.get(Dcard_story + '/' + Dcard_comments, params = params)
            comments_tmp = comments.json()
            if len(comments_tmp) == 0:
                break
            comments_j += comments_tmp
            params['after'] = comments_tmp[-1]['floor']
                            
        for comment in comments_j:
            if comment['hiddenByAuthor'] == False and comment['hidden'] == False:
                content = comment['content']
                # 找連結
                f_http = 0
                f_jpg = 0
                f_http = content.find('http', f_http)
                f_jpg = content.find('jpg', f_jpg)
                while f_http > 0 and f_jpg < 0:
                    f_http = content.find('http', f_http + 4)
                    #print('no')
                    if f_http < 0 :
                        break
                    
                while f_http > 0 and f_jpg > 0:                
                    c_url = content[f_http:f_jpg + 3]
                    link = c_url
                    print('comment', link)
                    aaa = link.split('//')[1]
                    bbb = aaa.split('/')[1] # get output file name! 
                    try:
                        urllib.request.urlretrieve(link, story_folder + '/' + bbb) # 儲存圖片 (url, route)
                        urllib.request.urlretrieve(link, all_folder + '/' + bbb) # 儲存圖片 (url, route)
                    except Exception as e:
                        print(e)
                    
                    f_http = content.find('http', f_http + 4)
                    f_jpg = content.find('jpg', f_jpg + 3)
                    

with open(ori_Dcard_folder + '/' + 'data.json', '+w') as f:
    json.dump(resp_j[0]['id'], f)
    print('update to', str(resp_j[0]['id']))       
        
        
  