import zipfile

with zipfile.ZipFile("data/dogs-vs-cats.zip", "r") as z:
    z.extractall(path="data")

with zipfile.ZipFile("data/train.zip", "r") as z:
    z.extractall(path="data")

with zipfile.ZipFile("data/test1.zip","r") as z:
    z.extractall(path="data")