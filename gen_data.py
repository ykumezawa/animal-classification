from PIL import Image
#glob ファイルの一覧を取得するためのパッケージ（パターン一致でファイル一覧を取得する）
import os, glob
import numpy as np
from sklearn import model_selection


classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)
image_size = 50

#画像の読み込み
X = []
Y = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i > 200:break
        image = Image.open(file)
        #imageをRGBの数字に変換をする
        image = image.convert('RGB')
        image = image.resize((image_size, image_size))
        #imageデータを数字の配列にして変数に渡す
        data = np.asarray(image)
        X.append(data)
        Y.append(index)
#TensorFlowが扱いやすいようにarray形式に変更をする。
X = np.array(X)
Y = np.array(Y)


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
#numpyのファイルをテキストファイルとして保存をする。
np.save("./animal.npy", xy)