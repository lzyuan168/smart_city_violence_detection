import os
import glob
import cv2
from PIL import Image
from tqdm import tqdm


def gen_list(input_dir, merge_size):
        merge_list = []
        img_list = []
        for vidPath in tqdm(sorted(glob.glob(input_dir + '/*'))):
            img_list.append(vidPath)
            if len(img_list) == merge_size:
                merge_list.append(img_list)
                img_list = []
        return merge_list


def merge_frames(input_dir, save_dir, merge_img_dim):
    """
    5 x 3 merge 
    if frames % 12 == 0:
        merge
    """
    r = merge_img_dim[0]
    c = merge_img_dim[1]
    merge_size = r * c

    os.makedirs(save_dir, exist_ok=True)
    for folderPath in tqdm(glob.glob(input_dir + '/*')):
        folderName = os.path.basename(folderPath).split('.')[0]
        merge_list = gen_list(folderPath, merge_size) 

        img = Image.open(merge_list[0][0])
        (w, h) = img.size
        result_w = w * c
        result_h = h * r
        result_img = Image.new('RGB', (result_w, result_h))

        for count, each in enumerate(merge_list):
            idx = 0
            for i in range(r):
                for j in range(c):
                    print(each[idx])
                    image = Image.open(each[idx])
                    result_img.paste(im=image, box=(j*w, i*h))
                    idx += 1

            merge_img_name = os.path.join(save_dir, "{}_merge_{}.jpg".format(folderName, count))
            result_img.save(merge_img_name)
            

# input_dir = "/home/li.zhiyuan/Desktop/Semester  3/Smart Cities/project/dataset/our dataset/Non-Violence frames"
# input_dir2 = "/home/li.zhiyuan/Desktop/Semester  3/Smart Cities/project/dataset/our dataset/violence frames (copy 1)"
# save_dir = "/home/li.zhiyuan/Desktop/Semester  3/Smart Cities/project/dataset/our dataset/Non-Violence frames merged"
# merge_img_dim = [5, 3]
# merge_frames(input_dir, save_dir, merge_img_dim)


def save_to_frame(input_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for path in tqdm(glob.glob(input_dir + '/*')):
        fname = os.path.basename(path).split('.')[0]
        os.makedirs(os.path.join(save_dir, fname), exist_ok=True)
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        while success:
            if count % 1 == 0:
                #print("count = {}".format(count))
                cv2.imwrite("{}/{}/{}-{}.jpg".format(save_dir, fname, fname, str(count).zfill(4)), image)     # save frame as JPEG file      
            success, image = vidcap.read()
            count += 1


# input_dir = "/home/li.zhiyuan/Desktop/Semester  3/Smart Cities/project/dataset/our dataset/Non-Violence Videos"
# save_dir = "/home/li.zhiyuan/Desktop/Semester  3/Smart Cities/project/dataset/our dataset/Non-Violence frames"
# save_to_frame(input_dir, save_dir)


def resize(imgPath, imgSize):
    data = []
    labels = []
    # loop over the image paths
    for imagePath in tqdm(imgPath[::]):
        # imagePath : file name ex) V_123
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2] # Violence / NonViolence

        # if the label of the current image is not part of of the labels
        # are interested in, then ignore the image
        if label not in LABELS:
            continue

        # load the image, convert it to RGB channel ordering, and resize
        # it to be a fixed 224x224 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imgSize, imgSize))

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels