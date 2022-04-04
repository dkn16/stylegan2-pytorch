import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import os
import numpy as np

from PIL import Image
from torch import dtype
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out


def prepare21cm(
    env, datapath, size=[64]
):
    # get the filenames in the datapath
    filenames=[]
    for root, dirs, files in os.walk(datapath, topdown=False):
        for name in files:
            filenames.append(os.path.join(root, name))

    #check the dimension of data
    datacache=np.load(filenames[0])
    if datacache.ndim!=3:
        print('Please check the dimension of 21cm cube, should be 3D')
        exit
    slices=np.shape(datacache)[0]

    #load the 21cm datacubes
    total=0
    for filename in filenames:
        datacache=np.load(filename).astype(np.float32)
        #in many cases we dont neet to do the normalization here, the 21cm signal is always less than 255, thus PIL can handle this
        #datacache=datacache/(3*np.std(datacache))
        for i in range(slices):
            key = f"{size[0]}-{str(total).zfill(6)}".encode("utf-8")
            with env.begin(write=True) as txn:
                txn.put(key, datacache[i])
            total += 1

    with env.begin(write=True) as txn:
        txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

# this function is specially designed for Xiaosheng's dataset
def prepareconditional21cm(
    env, size=[64,512]
):
    # get the filenames in the datapath. Here we can directly write the code
    filenames=[]
    paramfilename=[]
    for i in range(9998):
        filenames.append('/work/zxs/data_lo28/Idlt-'+str(i)+'.npy')
        paramfilename.append('/work/zxs/data_loo28/Idlt-'+str(i)+'-y.npy')

    #check the dimension of data
    datacache=np.load(filenames[0])
    if datacache.ndim!=3:
        print('Please check the dimension of 21cm cube, should be 3D')
        exit
    
    slices=np.shape(datacache)[0]
    width=np.shape(datacache)[1]
    length=np.shape(datacache)[2]
    widthstartpoint=int((size[0]-width)/2.)
    lengthstartpoint=int((size[1]-length)/2.)
    #load the 21cm datacubes
    total=0
    for i in range(9998):
        datacube=np.load(filenames[i]).astype(np.float32)
        datacache=datacube[:,widthstartpoint:widthstartpoint+size[0],lengthstartpoint:lengthstartpoint+size[1]]
        paramcache=np.load(paramfilename[i]).astype(np.float32)
        #in many cases we dont neet to do the normalization here, the 21cm signal is always less than 255, thus PIL can handle this
        #datacache=datacache/(3*np.std(datacache))
        for i in range(slices):
            key = f"{size[0]}-{str(total).zfill(7)}".encode("utf-8")
            key_label = f"{size[0]}-{str(total).zfill(7)}_label".encode("utf-8")
            with env.begin(write=True) as txn:
                txn.put(key, datacache[i])
                txn.put(key_label,paramcache)
            total += 1
        datacache=datacube.transpose(1,0,2)[:,widthstartpoint:widthstartpoint+size[0],lengthstartpoint:lengthstartpoint+size[1]]
        for i in range(slices):
            key = f"{size[0]}-{str(total).zfill(7)}".encode("utf-8")
            key_label = f"{size[0]}-{str(total).zfill(7)}_label".encode("utf-8")
            with env.begin(write=True) as txn:
                txn.put(key, datacache[i])
                txn.put(key_label,paramcache)
            total += 1

    with env.begin(write=True) as txn:
        txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="64,64",
        help="resolutions of images for the dataset",
    )
    #deleted the multiprocessing argument, because i'm too stupid to use multiprocessing
    #deleted the resampling argument, not needed anymore
    parser.add_argument("path", type=str, help="path to the image dataset")
    #a parameter for making conditional dataset
    parser.add_argument("--cond", type=bool, help="path to the image dataset",default=False)

    args = parser.parse_args()

    #resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    #resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = args.path

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        if args.cond:
            prepareconditional21cm(env,  size=sizes)
        else:
            prepare21cm(env, imgset,size=sizes)
