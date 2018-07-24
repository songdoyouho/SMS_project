from keras.models import load_model
model = load_model('model-dtaug.h5')

# load images and gather them
import os, cv2
import numpy as np
beauty_path = '/media/yopipi-ubuntu/SSD/ptt_beauty'

date_folders = os.listdir(beauty_path)
#print(date_folders, len(date_folders))
for i in range(1, len(date_folders)):
    # split the result images into two folders for visualize
    # map_characters = {0: 'yes', 1: 'no'}
    write_path = '/home/yopipi-ubuntu/桌面/yes_or_no'
    ok_path = os.path.join(write_path, 'ok')
    pred_yes_path = os.path.join(write_path, 'pred_yes')
    pred_yes_path = os.path.join(pred_yes_path, date_folders[i])
    pred_no_path = os.path.join(write_path, 'pred_no')
    pred_no_path = os.path.join(pred_no_path, date_folders[i])

    if not os.path.isdir(pred_no_path):
        try:
            os.mkdir(pred_no_path)
        except Exception as e:
            print(e)
            
        try:    
            os.mkdir(pred_yes_path)
        except Exception as e:
            print(e)

        test_folder = os.path.join(beauty_path, date_folders[i])
        print(test_folder)
        test_folders = os.listdir(test_folder)

        x_test, pics, all_pics_path, all_pic_path = [], [], [], []
        count = 1
        for folder in test_folders:
            folder_path = os.path.join(test_folder, folder)
            pics_path = os.listdir(folder_path)
            for pic in pics_path:
                pic_path = os.path.join(folder_path, pic)
                all_pics_path.append(pic_path)
                all_pic_path.append(pic)
                print(pic_path, count)
                temp = cv2.imread(pic_path)
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
                temp = cv2.resize(temp, (200, 200)).astype('float32') / 255.
                pics.append(temp)
                count = count + 1

        X_test = np.array(pics)

        # prediction!!!
        y_pred = model.predict_classes(X_test)

        for i in range(len(all_pics_path)):
            temp = cv2.imread(all_pics_path[i])
            label = y_pred[i]   
            if label == 0:
                tmp_path = os.path.join(pred_yes_path, all_pic_path[i])
                cv2.imwrite(tmp_path, temp)
                tmp_path = os.path.join(ok_path, all_pic_path[i])
                cv2.imwrite(tmp_path, temp)
            else:
                tmp_path = os.path.join(pred_no_path, all_pic_path[i])
                cv2.imwrite(tmp_path, temp)