import os

os.system("python 垂死挣扎_json.py")  # seed list需要修改的文件
os.system("python 单标签模型_初始.py")
os.system("python 垂死挣扎_单标签模型_继续.py")  # seed list需要修改的文件
os.system("python 垂死挣扎_多_初始.py")
os.system("python 垂死挣扎_多.py")  # seed list需要修改的文件
os.system("python 垂死挣扎_输出result_3.py")  # seed list需要修改的文件
os.system("python 垂死挣扎_聚类.py")  # seed list需要修改的文件
os.system("python 垂死挣扎_单标签结果.py")
