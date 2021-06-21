import glob
import imutils
import cv2
import numpy as np
import time


def _preprocessing(fileType):
    # init data and labels
    data = []
    labels = []
    # browse each link to the image
    for path in glob.glob(fileType):
        _, brand, fn = path.split('\\')
        # process image, using Canny
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edged = cv2.Canny(thresh1, 100, 200)

        # find contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        # norm contours
        cnts = imutils.grab_contours(cnts)
        # find max contours
        c = max(cnts, key=cv2.contourArea)
        # extract logo and resize image to 200x200
        (x, y, w, h) = cv2.boundingRect(c)
        logo = gray[y:y + h, x:x + w]
        logo = cv2.resize(logo, (200, 200))

        # create hog feature with each logo
        # init parameters
        cell_size = (8, 8)  # h x w in pixels
        block_size = (2, 2)  # h x w in cells
        nbins = 9  # number of orientation bins

        # compute the parameters passed to HOGDescriptor
        winSize = (logo.shape[1] // cell_size[1] * cell_size[1], logo.shape[0] // cell_size[0] * cell_size[0])
        blockSize = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
        # blockStride: Số bước di chuyển của block khi thực hiện chuẩn hóa histogram bước sau
        blockStride = (cell_size[1], cell_size[0])
        # Compute HOG descriptor
        hog = cv2.HOGDescriptor(_winSize=winSize,
                                _blockSize=blockSize,
                                _blockStride=blockStride,
                                _cellSize=cell_size,
                                _nbins=nbins)
        # create hog feature
        hog_feats = hog.compute(logo)
        # convert to one-dimensional array
        hog_feats = hog_feats.flatten()
        # update the data and labels
        data.append(hog_feats)
        labels.append(brand)
    return data, labels


# get data and labels form folders
data, labels = _preprocessing('CarLogo2/trainData/**/*.jpg')
data_test, labels_test = _preprocessing('CarLogo2/testData/**/*.jpg')

from sklearn.preprocessing import LabelEncoder


# create data for model
def _transform_data(data, labels):
    # create features X
    X = np.array(data)
    # encode labels
    le = LabelEncoder()
    le.fit(labels)
    # create labels
    y = le.transform(labels)
    return X, y


#
X_train, y_train = _transform_data(data, labels)
X_test, y_test = _transform_data(data_test, labels_test)

from sklearn import metrics

while True:
    print("_____Chon model_____")
    print("1. Linear Regression")
    print("2. Logistic Regression")
    print("3. Decision tree")
    print("4. Naive Bayes classifiers")
    print("5. SVM")
    print("0. Thoat")
    print("__________________")
    print("Nhap lua chon: ")
    choose = int(input())
    if choose == 0:
        break
    elif choose == 1:
        start = time.time()
        from sklearn.linear_model import LinearRegression

        linear = LinearRegression()

        linear.fit(X_train, y_train)
        y_tmp = linear.predict(X_test)
        y_pred = []
        for x in y_tmp:
            y_pred.append(round(x))
        end = time.time()
        print("Thoi gian chay modul: ", end - start)

    elif choose == 2:
        start = time.time()
        from sklearn.linear_model import LogisticRegression

        logis = LogisticRegression()
        logis.fit(X_train, y_train)
        y_pred = logis.predict(X_test)
        end = time.time()
        print("Thoi gian chay modul: ", end - start)
    elif choose == 3:
        start = time.time()
        from sklearn.tree import DecisionTreeRegressor

        regressor = DecisionTreeRegressor(random_state=0)

        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        end = time.time()
        print("Thoi gian chay modul: ", end - start)
    elif choose == 4:
        start = time.time()
        from sklearn.naive_bayes import GaussianNB

        gnb = GaussianNB()

        gnb.fit(X_train, y_train)

        # making predictions on the testing set
        y_pred = gnb.predict(X_test)
        end = time.time()
        print("Thoi gian chay modul: ", end - start)
    elif choose == 5:
        start = time.time()
        from sklearn.svm import SVC

        clf = SVC(kernel='linear')

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        end = time.time()
        print("Thoi gian chay modul: ", end - start)
    print("Ket qua thuc te cua nhan:")
    print(y_test)
    print("Ket qua du doan nhan:")
    print(y_pred)
    print("Ty le du doan chinh xac: ", metrics.accuracy_score(y_test, y_pred) * 100)
    print("\n")
