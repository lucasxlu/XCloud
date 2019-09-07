import os

from sklearn.model_selection import train_test_split

base_dir = '/path/to/your/SegDataset'

X = [_.split('.jpg')[0] for _ in os.listdir(os.path.join(base_dir, 'JPEGImages'))]
y = range(len(X))

X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2)

with open('./train.txt', mode='wt') as f:
    f.writelines('\n'.join(X_train))

with open('./val.txt', mode='wt') as f:
    f.writelines('\n'.join(X_test))
