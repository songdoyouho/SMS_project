from keras.models import load_model
model = load_model('model-dtaug.h5')

# load images and gather them
import os, cv2
import numpy as np
dcard_path = '/media/yopipi-ubuntu/SSD/Dcard'
count = 1
date_folders = os.listdir(dcard_path)
#print(date_folders, len(date_folders))
for i in range(0, len(date_folders) - 1):
    # split the result images into two folders for visualize
    # map_characters = {0: 'yes', 1: 'no'}
    write_path = '/home/yopipi-ubuntu/桌面/yes_or_no'
    ok_path = os.path.join(write_path, 'dcard_test')
    pred_yes_path = os.path.join(write_path, 'dcard_pred_yes')
    pred_yes_path = os.path.join(pred_yes_path, date_folders[i])
    pred_no_path = os.path.join(write_path, 'dcard_pred_no')
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

        test_folder = os.path.join(dcard_path, date_folders[i])
        #print(test_folder)
        test_folders = os.listdir(test_folder)

        x_test, pics, all_pics_path, all_pic_path = [], [], [], []
        
        for folder in test_folders:
            folder_path = os.path.join(test_folder, folder)
            pics_path = os.listdir(folder_path)
            count = count + 1
            for pic in pics_path:
                if pic[-3:] == 'jpg':
                    pic_path = os.path.join(folder_path, pic)
                    temp = cv2.imread(pic_path)
                    try:
                        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
                    except:
                        continue
                    temp = cv2.resize(temp, (200, 200)).astype('float32') / 255.
                    all_pics_path.append(pic_path)
                    all_pic_path.append(pic)
                    print(pic_path, count)
                    pics.append(temp)

        X_test = np.array(pics)

        # prediction!!!
        y_pred = model.predict_classes(X_test)

        for i in range(len(all_pics_path)):
            temp = cv2.imread(all_pics_path[i])
            label = y_pred[i]   
            if label == 0:
                tmp_path = os.path.join(pred_yes_path, all_pic_path[i])
                print(tmp_path)
                cv2.imwrite(tmp_path, temp)
                tmp_path = os.path.join(ok_path, all_pic_path[i])
                cv2.imwrite(tmp_path, temp)
            else:
                tmp_path = os.path.join(pred_no_path, all_pic_path[i])
                cv2.imwrite(tmp_path, temp)