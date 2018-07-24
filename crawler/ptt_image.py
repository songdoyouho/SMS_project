# -*- coding: utf-8 -*-

import requests
import time
from bs4 import BeautifulSoup
import os
import re
import urllib.request

today =time.strftime("%Y_%m_%d", time.localtime())
save_beauty = '/media/yopipi-ubuntu/SSD/ptt_beauty'
save_sex = '/media/yopipi-ubuntu/SSD/ptt_sex'
save_beauty = os.path.join(save_beauty, today)
save_sex = os.path.join(save_sex, today)

PTT_URL = 'https://www.ptt.cc'


def get_web_page(url):
    time.sleep(0.5)  # 每次爬取前暫停 0.5 秒以免被 PTT 網站判定為大量惡意爬取
    resp = requests.get(
        url=url,
        cookies={'over18': '1'}  # 是否滿18歲
    )
    if resp.status_code != 200:  # 確認可連上
        print('Invalid url:', resp.url)
        return None
    else:
        return resp.text


def get_articles(dom, date):
    soup = BeautifulSoup(dom, 'html.parser')

    # 取得上一頁的連結
    paging_div = soup.find('div', 'btn-group btn-group-paging')
    prev_url = paging_div.find_all('a')[1]['href']

    articles = []  # 儲存取得的文章資料
    divs = soup.find_all('div', 'r-ent')
    for d in divs:
        if d.find('div', 'date').string.strip() == date:  # 發文日期正確
            # 取得推文數
            push_count = 0
            if d.find('div', 'nrec').string:
                try:
                    push_count = int(d.find('div', 'nrec').string)  # 轉換字串為數字
                except ValueError:  # 若轉換失敗，不做任何事，push_count 保持為 0
                    pass

            # 取得文章連結及標題
            if d.find('a'):  # 有超連結，表示文章存在，未被刪除
                href = d.find('a')['href']
                title = d.find('a').string
                articles.append({
                    'title': title,
                    'href': href,
                    'push_count': push_count
                })
    return articles, prev_url


def parse(dom):
    soup = BeautifulSoup(dom, 'html.parser')
    links = soup.find(id='main-content').find_all('a')
    img_urls = []
    for link in links:
        if re.match(r'^https?://(i.)?(m.)?imgur.com', link['href']):
            img_urls.append(link['href'])
    return img_urls


def save(img_urls, title, anno):
    if img_urls:
        try:
            dname = title.strip()  # 用 strip() 去除字串前後的空白
            if dname[0] == 'R':
                dname = dname[3:]
                
            dname = anno + '/' + dname
            print(dname)
            os.makedirs(dname)
            for img_url in img_urls:
                if img_url.split('//')[1].startswith('m.'):
                    img_url = img_url.replace('//m.', '//i.')
                if not img_url.split('//')[1].startswith('i.'):
                    img_url = img_url.split('//')[0] + '//i.' + img_url.split('//')[1]
                if not img_url.endswith('.jpg'):
                    break
                    #img_url += '.jpg'
                fname = img_url.split('/')[-1]
                print(img_url,  os.path.join(dname, fname))
                urllib.request.urlretrieve(img_url, os.path.join(dname, fname))
        except Exception as e:
            print(e)
            '''print('檔案已存在! 尋找下一個...')'''


if __name__ == '__main__':
    from datetime import timedelta, datetime  
    yesterday = datetime.today() + timedelta(-1)
    date = yesterday.strftime('%m/%d').lstrip('0')
    print(date)
    '''
    date = time.strftime("%m/%d").lstrip('0')  # 取得今天日期, 去掉開頭的 '0' 以符合 PTT 網站格式
    date_length = len(date)
    if date_length == 4:
        if date[3] == '0':
            date_ten = int(date[2])
            date_go = int(date[3])
            yesterday = date[0] + date[1] + str(date_ten - 1) + '9'
            two_date = [yesterday, date]
        else:
            date_go = int(date[3])
            yesterday = date[0] + date[1] + date[2] + str(date_go - 1)
            two_date = [yesterday, date]
            
    if date_length == 5:
        if date[4] == '0':
            date_ten = int(date[3])
            date_go = int(date[4])
            yesterday = date[0] + date[1] + date[2] + str(date_ten - 1) + '9'
            two_date = [yesterday, date]
        else:
            date_go = int(date[4])
            yesterday = date[0] + date[1] + date[2] + date[3] + str(date_go - 1)
            two_date = [yesterday, date]
    
    for date in two_date:
    '''
    beauty_current_page = []
    beauty_current_page = get_web_page(PTT_URL + '/bbs/Beauty/index.html')  # 加上表特版的網址
    # 分析表特版
    if beauty_current_page:
        print(date, 'beauty')
        articles = []  # 全部的今日文章       
        current_articles, prev_url = get_articles(beauty_current_page, date)  # 目前頁面的今日文章
        while not current_articles:
            beauty_current_page = get_web_page(PTT_URL + prev_url)
            current_articles, prev_url = get_articles(beauty_current_page, date)
            
        while current_articles:  # 若目前頁面有今日文章則加入 articles，並回到上一頁繼續尋找是否有今日文章
            articles += current_articles
            beauty_current_page = get_web_page(PTT_URL + prev_url)
            current_articles, prev_url = get_articles(beauty_current_page, date)

        # 已取得文章列表，開始進入各文章讀圖
        for article in articles:
            print('Processing', article['title'], '推文數 :',article['push_count'])
            page = get_web_page(PTT_URL + article['href'])
            if page:
                img_urls = parse(page)
                save(img_urls, article['title'], save_beauty)
                article['num_image'] = len(img_urls)

    # 分析西斯版
    sex_current_page = []
    sex_current_page = get_web_page(PTT_URL + '/bbs/Sex/index.html')  # 加上西斯版的網址
    if sex_current_page:
        print(date, 'sex')
        articles = []  # 全部的今日文章        
        current_articles, prev_url = get_articles(sex_current_page, date)  # 目前頁面的今日文章
        while not current_articles:
            sex_current_page = get_web_page(PTT_URL + prev_url)
            current_articles, prev_url = get_articles(sex_current_page, date)
    
        while current_articles:  # 若目前頁面有今日文章則加入 articles，並回到上一頁繼續尋找是否有今日文章
            articles += current_articles
            sex_current_page = get_web_page(PTT_URL + prev_url)
            current_articles, prev_url = get_articles(sex_current_page, date)

        # 已取得文章列表，開始進入各文章讀圖
        for article in articles:
            print('Processing', article['title'], '推文數 :', article['push_count'])
            page = get_web_page(PTT_URL + article['href'])
            if page:
                img_urls = parse(page)
                save(img_urls, article['title'], save_sex)
                article['num_image'] = len(img_urls)

        # 儲存文章資訊
        '''with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, sort_keys=True, ensure_ascii=False)'''