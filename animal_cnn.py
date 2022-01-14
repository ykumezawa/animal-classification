from gen_data import X_train
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import keras



classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)
image_size = 50

#メインの関数を定義する
def main():
    #ファイルからデータを配列に読み込む
    X_train, X_test, y_train, y_test = np.load("./animal.npy",allow_pickle=True)
    #データの正規化をする（=256で割る）
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    # one-hot-vector:正解値は１、他は0 のベクトルに変換する。（そのためにto_categoricalを利用する。）
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)


def model_train(X, y):
    model = Sequential()
    #畳み込み結果が同じサイズになるようにピクセルを左右に足す。（padding="same"）
    model.add(Conv2D(32,(3,3), padding='same', input_shape = X.shape[1:]))
    #Activationfunction reluは正の値のみ反映をさせるという活性化関数
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    #pool_size-(2,2)は一番大きい値を取り出すということ（より特徴を際立たせる。）
    model.add(MaxPooling2D(pool_size=(2,2)))
    #Dropout(0,25) 25%をdropoutさせることでモデルの偏りを少なくする。
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #ここからは全結合するためのコード
    #データを一列に並べる。(Flatten)
    model.add(Flatten())
    #Dense:全結合層
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #最後の出力層のノードが3つなのでDense(3)
    model.add(Dense(3))
    model.add(Activation('softmax'))

    #最適化の手法の宣言をしている。
    #optimizer.rmspropとはトレーニング時の更新アルゴリズム
    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    #loss: 損失関数（正解と推定値の誤差）
    model.compile(loss = 'categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    #batch_sizeは一エポックで使用するデータの数
    #nb_epoch(number of epoch)epochを何回行うのか。
    model.fit(X, y, batch_size = 32, epochs = 100)

#モデルを保存している。
    model.save('./amimal_css.h5')

    return model
# テストの関数でモデルを使えるように下記の記述をしている。
def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose =1)
    print('Test loss:', scores[0])
    print('Test Accuracy:', scores[1])

#このプログラムが直接Pythonから呼ばれた時だけmainを実行する。
#そうでなければ、各関数を引用して使うことができる
if __name__ == "__main__":
    main()