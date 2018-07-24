from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
from linebot.models import *

from instaloader import Profile

import os, random, json, requests, secret, instaloader, urllib, threading, time

keys = secret.keys()

beauty_img_path = '/home/yopipi-ubuntu/桌面/yes_or_no/ok'
dcard_img_path = '/home/yopipi-ubuntu/桌面/yes_or_no/dcard_test'
ig_save_path = '/home/yopipi-ubuntu/桌面/yes_or_no/ig'
beauty_img_list = os.listdir(beauty_img_path)
dcard_img_list = os.listdir(dcard_img_path)
now_path = os.getcwd()

imgur_url = 'https://imgur.com/'

app = Flask(__name__)

line_bot_api = LineBotApi(keys.line_api)
handler = WebhookHandler(keys.line_secret)

@app.route("/", methods=['POST'])
def index():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    json_line = request.get_json()
    json_line = json.dumps(json_line)
    global decoded
    decoded = json.loads(json_line)
    #print(decoded)
    print(decoded["events"][0]["message"]["text"])
    # save userid and groupid in data.json
    if decoded["events"][0]["source"]["type"] == "user":
        try:
            print(decoded["events"][0]["source"]["userId"], decoded["events"][0]["message"]["text"])
            with open(now_path + '/' + 'data.json', 'r') as f:
                data = json.load(f)
                f.close()
            with open(now_path + '/' + 'data.json', 'w') as f:
                if decoded["events"][0]["source"]["userId"] in data['userid']:
                    print('in the userid array')
                else:
                    data['userid'].append(decoded["events"][0]["source"]["userId"])
                json.dump(data, f)
                f.close()

        except Exception as e:
            print(e) 
    elif decoded["events"][0]["source"]["type"] == "group":
        try:
            print(decoded["events"][0]["source"]["groupId"], decoded["events"][0]["message"]["text"])
            with open(now_path + '/' + 'data.json', 'r') as f:
                data = json.load(f)
                f.close()
            with open(now_path + '/' + 'data.json', 'w') as f:
                if decoded["events"][0]["source"]["groupId"] in data['groupid']:
                    print('in the groupid array')
                else:
                    data['groupid'].append(decoded["events"][0]["source"]["groupId"])
                json.dump(data, f)
                f.close()
        except Exception as e:
            print(e)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # 隨機抽我的表特版圖庫
    if event.message.text==u"抽":
        length = len(beauty_img_list) - 1
        img = beauty_img_list[random.randint(0,length)]
        url = imgur_url + img
        image_message = ImageSendMessage(
            original_content_url=url,
            preview_image_url=url
        )
        line_bot_api.reply_message(
            event.reply_token, image_message)
    # 隨機抽我的dcard圖庫
    elif event.message.text==u"卡":
        length = len(dcard_img_list) - 1
        img = dcard_img_list[random.randint(0,length)]
        url = imgur_url + img
        image_message = ImageSendMessage(
            original_content_url=url,
            preview_image_url=url
        )
        line_bot_api.reply_message(
            event.reply_token, image_message)
    # 推薦 ig 帳號
    elif event.message.text[:2]==u"推薦":
        # 分男女

        ig_id = event.message.text[3:]

        # Get instance
        L = instaloader.Instaloader(post_metadata_txt_pattern= '', download_videos=False, download_video_thumbnails=False, download_geotags=False, download_comments=False, compress_json=False)

        # 找前100篇圖
        profile = Profile.from_username(L.context, ig_id)
        all_likes = []
        all_urls = []
        count = 1
        for post in profile.get_posts():
            all_likes.append(post.likes)
            all_urls.append(post.url)
            #print(count)
            count += 1
            if count == 100: # 爬 100 張照片
                break

        try:
            person_path = os.path.join(ig_save_path, ig_id)
            os.mkdir(person_path)
        except Exception as e:
            print(e)

        sorted_urls = [all_urls for _,all_urls in sorted(zip(all_likes,all_urls), reverse=True)]
        # 找按讚數前50的載
        download_urls = sorted_urls[:50]
        count = 1
        for url in download_urls:
            print(url)
            final_path = os.path.join(person_path, str(count)+'.jpg')
            urllib.request.urlretrieve(url, final_path)
            count += 1
        
        # 把網址存在json裡給line bot用
        with open(os.path.join(person_path, 'url.json'), 'w') as f:
            json.dump(download_urls, f)  

        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text='圖片已抓取'))  
    elif event.message.text==u"更新":
         # Get instance
        L = instaloader.Instaloader(post_metadata_txt_pattern= '', download_videos=False, download_video_thumbnails=False, download_geotags=False, download_comments=False, compress_json=False)
        all_person = os.listdir(ig_save_path)
        for person in all_person:
            person_path = os.path.join(ig_save_path, person)
            final_path = os.path.join(ig_save_path, person)
            # 找前100篇圖
            profile = Profile.from_username(L.context, person)
            all_likes = []
            all_urls = []
            count = 1
            for post in profile.get_posts():
                all_likes.append(post.likes)
                all_urls.append(post.url)
                #print(count)
                count += 1
                if count == 100: # 爬 100 張照片
                    break

            sorted_urls = [all_urls for _,all_urls in sorted(zip(all_likes,all_urls), reverse=True)]
            # 找按讚數前50的載
            download_urls = sorted_urls[:50]
            count = 1
            for url in download_urls:
                print(url)
                final_path = os.path.join(person_path, str(count)+'.jpg')
                urllib.request.urlretrieve(url, final_path)
                count += 1

             # 把網址存在json裡給line bot用
            with open(os.path.join(person_path, 'url.json'), 'w') as f:
                json.dump(download_urls, f)  

        line_bot_api.push_message('U4a163602a7d66b0494cc38f4824d4d44', TextSendMessage(text='圖片已抓取'))  
    # 隨機抽 ig 圖庫裡的圖
    elif event.message.text.lower()==u"ig":
        person_folders = os.listdir(ig_save_path)
        global ran_folder
        ran_folder = person_folders[random.randint(0,len(person_folders) - 1)]
        folder = os.path.join(ig_save_path,ran_folder)
        image_list = os.listdir(folder)
        ran_image_num = random.randint(0,len(image_list) - 2)
        print(folder, ran_image_num)
        with open(os.path.join(folder, 'url.json'), 'r') as f:
            data = json.load(f)
            url = data[ran_image_num]
            image_message = ImageSendMessage(
            original_content_url=url,
            preview_image_url=url
            )
            line_bot_api.reply_message(event.reply_token, image_message)
    elif event.message.text.lower()==u"男ig":
        print("test")
    # 回答上一張圖是哪個 ig 帳號
    elif event.message.text==u"誰":
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=ran_folder))
    # 指定抽某個 ig 帳號的圖出來
    elif event.message.text[:1]==u"找":
        direct_folder = event.message.text[2:]
        folder = os.path.join(ig_save_path,direct_folder)
        image_list = os.listdir(folder)
        ran_image_num = random.randint(0,len(image_list) - 2)
        print(folder, ran_image_num)
        with open(os.path.join(folder, 'url.json'), 'r') as f:
            data = json.load(f)
            url = data[ran_image_num]
            image_message = ImageSendMessage(
            original_content_url=url,
            preview_image_url=url
            )
            line_bot_api.reply_message(event.reply_token, image_message)
    elif event.message.text[:1]==u"投":
        ig_id = event.message.text[2:]
        if os.path.isfile('vote.json'):
            with open('vote.json', 'r') as r:
                data = json.load(r)
                r.close()
                tmp_dict = {"userid" : event.source.user_id, "igid" : ig_id}
                if tmp_dict in data:
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="你已經投過摟！"))
                else:
                    data.append(tmp_dict)
                    with open('vote.json', 'w') as w:
                        json.dump(data, w)
                        w.close()
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="你已經投票！"))
        else:
            with open('vote.json', 'w') as w:
                data = [{"userid" : event.source.user_id, "igid" : ig_id}]
                json.dump(data, w)
                w.close()
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="你已經投票！"))
    # 顯示 ig 功能
    elif event.message.text==u"ig功能":
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text='輸入ig即可抽ig妹子，想推薦妹子請打：推薦（空格）ig的id。 \n例如：推薦 xxx \n找特定帳號的圖，請打：找（空格）ig的id \n想知道上一張圖是誰請打：誰 \n如果你覺得哪個帳號的圖你不喜歡，打：投（空格）ig的id，投他一票，我會視票數決定是否從圖庫刪除。'))     
    # 顯示虛擬幣功能
    elif event.message.text==u"虛擬幣功能":
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text='以下請注意中間要加空白鍵！ \n輸入：check（空格）xxxyyy 可查詢匯率。 \n例如：check（空格）ethbtc。')) 
        #line_bot_api.reply_message(
            #event.reply_token, TextSendMessage(text='以下請注意中間要加空白鍵！ \n輸入：check（空格）xxxyyy 可查詢匯率。 \n例如：check（空格）ethbtc。 \n 輸入：order（空格）xxxyyy（空格）>（空格）500 可設定到價通知。 \n例如：order（空格）btcusdt（空格）>（空格）8000。 \n目前到價通知測試中喔！')) 
    # 查詢幣安的當下虛擬貨幣匯率
    elif event.message.text[:5]==u"check":
        binance_request = requests.get("https://api.binance.com/api/v1/ticker/24hr")
        binance_info = json.loads(str(binance_request.content, encoding = "utf-8"))
        for info in binance_info:
            if event.message.text[6:].lower()==info['symbol'].lower():
                line_bot_api.reply_message(
                    event.reply_token, TextSendMessage(text=info['symbol'] + ":" + info['lastPrice']))
    elif event.message.text[:3]==u"給作者":
        user_message = event.message.text[4:]
        line_bot_api.push_message(keys.my_user_id,TextSendMessage(text=user_message))
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=' 已送達！'))
    '''
    # 設定到價通知
    elif event.message.text[:5]==u"order":
        input_order = event.message.text[6:]
        split_order = input_order.split(' ')
        split_order.append(event.source.user_id)
        if len(split_order) != 4:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text='我覺得不行！ \n請確認格式～'))
        else:
            if os.path.isfile('order.json'):
                with open('order.json', 'r') as r:
                    old_order = json.load(r)
                    r.close()
                    with open('order.json', 'w') as w:
                        old_order.append(split_order)
                        json.dump(old_order, w)
                        w.close()
            else:
                with open('order.json', 'w') as w:
                    old_order = [split_order]
                    json.dump(old_order, w)
                    w.close()
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text='roger that!'))
            flagg = True
            while flagg:
                with open('binance_info.json', 'r') as r:
                    binance_info = json.load(r)
                    r.close()
                for info in binance_info:
                    if split_order[0].lower()==info['symbol'].lower():
                        if split_order[1] == ">" or split_order[1] == ">=":
                            if info['lastPrice'] >= split_order[2]:
                                line_bot_api.push_message(event.source.user_id,TextSendMessage(text=input_order + "\n已達成！"))
                                if split_order in old_order:
                                    old_order.remove(split_order)
                                    with open('order.json', 'w') as w:
                                        json.dump(old_order, w)
                                        w.close()
                                flagg = False
                        elif split_order[1] == ">" or split_order[1] == "<=":
                            if info['lastPrice'] <= split_order[2]:
                                line_bot_api.push_message(event.source.user_id,TextSendMessage(text=input_order + "\n已達成！"))
                                if split_order in old_order:
                                    old_order.remove(split_order)
                                    with open('order.json', 'w') as w:
                                        json.dump(old_order, w)
                                        w.close()
                                flagg = False    
                time.sleep(30)
    # 重設到價通知        
    elif event.message.text[:5]==u"reset":
        print("reset order")
        with open('order.json', 'r') as r:
            old_order = json.load(r)
            r.close()
        print(old_order)
        flagg = True
        while flagg:
            with open('binance_info.json', 'r') as r:
                binance_info = json.load(r)
                r.close()
            for info in binance_info:
                for order in old_order:
                    if order[0].lower()==info['symbol'].lower():
                        if order[1] == ">" or order[1] == ">=":
                            if info['lastPrice'] >= order[2]:
                                input_order = order[0] + order[1] + order[2]
                                line_bot_api.push_message(order[3],TextSendMessage(text=input_order + "\n已達成！"))
                                if order in old_order:
                                    old_order.remove(order)
                                    with open('order.json', 'w') as w:
                                        json.dump(old_order, w)
                                        w.close()
                                if len(old_order) > 0 :
                                    continue
                                else:
                                    flagg = False
                        elif order[1] == "<" or order[1] == "<=":
                            if info['lastPrice'] <= order[2]:
                                input_order = order[0] + order[1] + order[2]
                                line_bot_api.push_message(order[3],TextSendMessage(text=input_order + "\n已達成！"))
                                if order in old_order:
                                    old_order.remove(order)
                                    with open('order.json', 'w') as w:
                                        json.dump(old_order, w)
                                        w.close()
                                if len(old_order) > 0:
                                    continue
                                else:
                                    flagg = False    
            time.sleep(30)
    '''

if __name__ == "__main__":
    app.run(host='127.0.0.1', port= 8080)