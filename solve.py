
from Input.process.solve import _shape
import numpy as np

X_train = np.loadtxt("F:/NhanDienLogo/Input/Files/train_X.txt").reshape(_shape()[0])
y_train = np.loadtxt("F:/NhanDienLogo/Input/Files/train_y.txt").reshape(_shape()[1])

X_test = np.loadtxt("F:/NhanDienLogo/Input/Files/test_X.txt").reshape(_shape()[2])
y_test = np.loadtxt("F:/NhanDienLogo/Input/Files/test_y.txt").reshape(_shape()[3])

def _data():
    return [X_train, y_train, X_test, y_test]
