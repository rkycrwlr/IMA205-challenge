import torch
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

from scipy import ndimage
from skimage.morphology import binary_opening, disk
from skimage.color import rgb2lab, rgb2hsv
from unet import UNet

from constants import WORKING_DIR, DEVICE, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT

def fill_holes(img):
    """
    Function to fill holes in a segmentation map
    """
    img_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill, mask, (0,0), 255)
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    return img | img_floodfill_inv

def keep_biggest_CC(img):
    """
    Function to only keep the connected component with the biggest area in a segmentation map
    """
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=4)
    l = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)]
    if l:
        max_label, _ = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
        img[output != max_label] = 0
    else:
        img[:] = 0
    return img

def make_pred(model, img):
    """
    Function to preprocess the output of the UNet
    """
    model.eval()
    img = img.unsqueeze(0).to(DEVICE)
    y = model(img).squeeze()
    y_fg = torch.sigmoid(y[0]).cpu().detach().numpy()
    y_bg = torch.sigmoid(y[1]).cpu().detach().numpy()
    r = (y_fg > y_bg) * 255
    r = r.astype(np.uint8)
    r = keep_biggest_CC(r)
    r = fill_holes(r)
    return r

def feature_extraction(metadata, img_dir, out_file):
    """
    This function extracts features from the lesions following various research articles:

    From the article https://www.sciencedirect.com/science/article/abs/pii/S0933365713001589, we extract :
        - f1, f2 (Section 2.3.1)
        - f3, f4 (Section 2.3.3)
        - f5, f6, f7 (Section 2.3.5)
        - f31, f32, f33 (Section 2.3.6)
        - f8, f9, f10 (Section 2.3.8)
    
    Following https://ieeexplore.ieee.org/document/918473, we extract the mean, variance, 5-percentile and 95-percentile of Hue and Value channels of the image in HSV :
        - f23, f24, f25, f26, f27, f28, f29, f30
    
    We decide to extract this four characteristics for RGB channels as well:
        - f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22
    """

    df = pd.read_csv(metadata)
    df_features = df.copy()

    t = transforms.Compose([
        transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()
    ])

    f1, f2, f3, f4, f5, f6, f7, = [], [], [], [], [], [], []
    f8, f9, f10, f11, f12, f13, f14, = [], [], [], [], [], [], []
    f15, f16, f17, f18, f19, f20, f21 = [], [], [], [], [], [], []
    f22, f23, f24, f25, f26, f27, f28 = [], [], [], [], [], [], []
    f29, f30, f31, f32, f33 = [], [], [], [], []

    for idx in tqdm(range(len(df['ID'].values))):
        img_path = df['ID'].values[idx]
        filename = img_dir + img_path + '.jpg'

        # Open rgb image at the good ratio
        img_rgb = Image.open(filename)
        img_rgb.thumbnail((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))

        # Convert it to LAB for later
        img_lab = rgb2lab(img_rgb)
        original_size = img_rgb.size
        img_rz = t(img_rgb)

        # Segmentation of the lesion with good ratio
        seg_rz = make_pred(unet, img_rz)
        seg = Image.fromarray(seg_rz).resize(original_size)
        seg = np.array(seg) > 0

        # Gray value masked image
        img_g = np.array(img_rgb.convert('L'))
        img_g_msk = img_g * seg
        img_rgb = np.array(img_rgb)
        
        x,y = np.where(img_g_msk > 0)
        # If the segmented area it too small return 0 for all the features
        if len(x) < 10:
            f1.append(0);f2.append(0);f3.append(0);f4.append(0);f5.append(0);f6.append(0);f7.append(0)
            f8.append(0);f9.append(0);f10.append(0);f11.append(0);f12.append(0);f13.append(0);f14.append(0)
            f15.append(0);f16.append(0);f17.append(0);f18.append(0);f19.append(0);f20.append(0);f21.append(0)
            f22.append(0);f23.append(0);f24.append(0);f25.append(0);f26.append(0);f27.append(0);f28.append(0)
            f29.append(0);f30.append(0);f31.append(0);f32.append(0);f33.append(0)
            continue

        # Pad the mask image to make it of size (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
        b1, b2, b3, b4 = np.min(x), np.max(x), np.min(y), np.max(y)
        img_g_msk_sq = np.zeros_like(seg_rz, np.int32)
        h_off = (((INPUT_IMAGE_HEIGHT-1) - b2) - b1)//2
        w_off = (((INPUT_IMAGE_WIDTH-1) - b4) - b3)//2
        img_g_msk_sq[b1+h_off:b2+h_off+1, b3+w_off:b4+w_off+1] = img_g_msk[b1:b2+1, b3:b4+1]
        
        # Extraction of features f5, f6, f7
        img_lab_msk = np.array(img_lab) * seg[:,:,np.newaxis]
        lab_values = img_lab_msk[np.sum(img_lab_msk,axis=2)!=0]
        H, _ = np.histogramdd(lab_values[np.random.choice(np.arange(len(lab_values)), size=10000)], bins=50)

        # Extraction of features f1, f2
        averageDS = float("inf")
        for angle in range(0,180,10):        
            
            img_r = np.clip(ndimage.rotate(img_g_msk_sq, angle, output=np.int32 ,reshape=False), 0, 255)
            tot = np.sum(img_r > 0)
            if tot == 0:
                continue
            
            # We extract the four regions described in the article
            img_A12 = img_r[:,:INPUT_IMAGE_WIDTH//2]
            img_A34 = img_r[:,INPUT_IMAGE_WIDTH//2:INPUT_IMAGE_WIDTH]
            img_A14 = img_r[:INPUT_IMAGE_HEIGHT//2,:]
            img_A23 = img_r[INPUT_IMAGE_HEIGHT//2:INPUT_IMAGE_HEIGHT, :]

            DS1 = np.sum(np.abs(img_A12-np.fliplr(img_A34)))
            DS2 = np.sum(np.abs(img_A14-np.flipud(img_A23)))
            
            DS1 /= tot
            DS2 /= tot

            if((DS1+DS2)/2 < averageDS):
                minDS1 = DS1
                minDS2 = DS2
                averageDS = (DS1+DS2)/2

        # Extraction of features f3, f4
        x,y = np.where(img_g_msk_sq > 0)
        x,y = x.mean(), y.mean()
        values = img_g_msk_sq[img_g_msk_sq > 0]
        percentiles = np.percentile(values, np.arange(10,91,10))
        v = []
        for p in percentiles:
            xt,yt = np.where(img_g_msk_sq > p)
            xt,yt = xt.mean(), yt.mean()
            v.append(np.sqrt((x-xt)**2 + (y-yt)**2))
        v = np.array(v)
        v /= np.sqrt(tot/np.pi)
        
        # Extraction of features f31, f32, f33
        msk = (img_g_msk_sq > 0) * 1
        contours, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        border = contours[0]
        target_pixels_inner = int(0.7 * np.sum(msk))
        center = np.array(x,y)
        pixel_step = 2
        while True:
            # Move all border points towards the center of mass
            for point in border:
                dir = center - np.array(point[0])
                if np.linalg.norm(dir) < 1e-3:
                    continue
                dir = (dir/np.linalg.norm(dir)) * pixel_step
                point[0][0] += int(dir[0])
                point[0][1] += int(dir[1])

            # Create a mask for the new outer region
            new_inner_mask = np.zeros_like(msk, dtype=np.int32)
            cv2.drawContours(new_inner_mask, (np.array(border),), 0, 1, -1)
            
            # Update the outer and inner parts
            inner_part = new_inner_mask
            outer_part = ((msk - new_inner_mask) > 0)*1

            # Check if the target number of pixels is reached
            if np.sum(inner_part) <= target_pixels_inner:
                break

        img_rgb_msk = np.zeros((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 3), dtype=np.int32)
        img_rgb_msk[b1+h_off:b2+h_off, b3+w_off:b4+w_off] = (img_rgb * seg[:,:,np.newaxis])[b1:b2, b3:b4] 
        img_lab_s = rgb2lab(img_rgb_msk/255)
        img_lab_s_inner = img_lab_s * inner_part[:,:,np.newaxis]
        img_lab_s_outer = img_lab_s * outer_part[:,:,np.newaxis]

        L_d = np.abs(img_lab_s_inner[:,:,0][img_lab_s_inner[:,:,0]!=0].mean() - img_lab_s_outer[:,:,0][img_lab_s_outer[:,:,0]!=0].mean())
        a_d = np.abs(img_lab_s_inner[:,:,1][img_lab_s_inner[:,:,1]!=0].mean() - img_lab_s_outer[:,:,1][img_lab_s_outer[:,:,1]!=0].mean())
        b_d = np.abs(img_lab_s_inner[:,:,2][img_lab_s_inner[:,:,2]!=0].mean() - img_lab_s_outer[:,:,2][img_lab_s_outer[:,:,2]!=0].mean())

        # Extraction of features f8, f9, f10
        percentiles = np.percentile(values, [25, 50, 75])
        n_components = []
        for p in percentiles:
            ret, thresh = cv2.threshold(img_g_msk_sq.astype(np.uint8), p, 255, cv2.THRESH_BINARY)
            im_p = binary_opening(thresh, disk(5)).astype(np.uint8)
            out = cv2.connectedComponents(im_p)
            n_components.append(float(out[0]-1))
        
        # Extraction of features f11, ..., f30
        img_hsv = rgb2hsv(img_rgb/255)
        avg_h_out = (img_hsv*(1-seg[:,:,np.newaxis]))[:,:,2].mean()
        unique, count = np.unique(img_hsv[:,:,0], return_counts= True)
        img_hsv[:,:,0] -= unique[np.argmax(count)]
        img_hsv[:,:,2] -= avg_h_out
        img_hsv_msk = img_hsv * seg[:,:,np.newaxis]

        r_lesion_values = img_rgb_msk[:,:,0][img_rgb_msk[:,:,0]>0]/255
        g_lesion_values = img_rgb_msk[:,:,1][img_rgb_msk[:,:,1]>0]/255
        b_lesion_values = img_rgb_msk[:,:,2][img_rgb_msk[:,:,2]>0]/255
        h_lesion_values = img_hsv_msk[:,:,0][img_hsv_msk[:,:,0]!=0]
        v_lesion_values = img_hsv_msk[:,:,2][img_hsv_msk[:,:,2]!=0]

        f1.append(minDS1)
        f2.append(minDS2)
        f3.append(v.mean())
        f4.append(v.std())
        f5.append(H[H>0].mean())
        f6.append(H[H>0].std())
        f7.append(np.sum(H>0)/(50*50*50))
        f8.append(n_components[0])
        f9.append(n_components[1])
        f10.append(n_components[2])
        f11.append(np.percentile(r_lesion_values, 5)  if r_lesion_values.size>0 else 0)
        f12.append(np.percentile(r_lesion_values, 95) if r_lesion_values.size>0 else 0)
        f13.append(np.mean(r_lesion_values)           if r_lesion_values.size>0 else 0)
        f14.append(np.var(r_lesion_values)            if r_lesion_values.size>0 else 0)
        f15.append(np.percentile(g_lesion_values, 5)  if g_lesion_values.size>0 else 0)
        f16.append(np.percentile(g_lesion_values, 95) if g_lesion_values.size>0 else 0)
        f17.append(np.mean(g_lesion_values)           if g_lesion_values.size>0 else 0)
        f18.append(np.var(g_lesion_values)            if g_lesion_values.size>0 else 0)
        f19.append(np.percentile(b_lesion_values, 5)  if b_lesion_values.size>0 else 0)
        f20.append(np.percentile(b_lesion_values, 95) if b_lesion_values.size>0 else 0)
        f21.append(np.mean(b_lesion_values)           if b_lesion_values.size>0 else 0)
        f22.append(np.var(b_lesion_values)            if b_lesion_values.size>0 else 0)
        f23.append(np.percentile(h_lesion_values, 5)  if h_lesion_values.size>0 else 0)
        f24.append(np.percentile(h_lesion_values, 95) if h_lesion_values.size>0 else 0)
        f25.append(np.mean(h_lesion_values)           if h_lesion_values.size>0 else 0)
        f26.append(np.var(h_lesion_values)            if h_lesion_values.size>0 else 0)
        f27.append(np.percentile(v_lesion_values, 5)  if v_lesion_values.size>0 else 0)
        f28.append(np.percentile(v_lesion_values, 95) if v_lesion_values.size>0 else 0)
        f29.append(np.mean(v_lesion_values)           if v_lesion_values.size>0 else 0)
        f30.append(np.var(v_lesion_values)            if v_lesion_values.size>0 else 0)
        f31.append(L_d)
        f32.append(a_d)
        f33.append(b_d)

    df_features["f1"] = f1
    df_features["f2"] = f2
    df_features["f3"] = f3
    df_features["f4"] = f4
    df_features["f5"] = f5
    df_features["f6"] = f6
    df_features["f7"] = f7
    df_features["f8"] = f8
    df_features["f9"] = f9
    df_features["f10"] = f10
    df_features["f11"] = f11
    df_features["f12"] = f12
    df_features["f13"] = f13
    df_features["f14"] = f14
    df_features["f15"] = f15
    df_features["f16"] = f16
    df_features["f17"] = f17
    df_features["f18"] = f18
    df_features["f19"] = f19
    df_features["f20"] = f20
    df_features["f21"] = f21
    df_features["f22"] = f22
    df_features["f23"] = f23
    df_features["f24"] = f24
    df_features["f25"] = f25
    df_features["f26"] = f26
    df_features["f27"] = f27
    df_features["f28"] = f28
    df_features["f29"] = f29
    df_features["f30"] = f30
    df_features["f31"] = f31
    df_features["f32"] = f32
    df_features["f33"] = f33
    df_features.to_csv(out_file, index=False)

unet = UNet().to(DEVICE)
unet.load_state_dict(torch.load("unet_seg_200e.pth", map_location=DEVICE)) # We load the model previously trained

# We extract all the features and write them in a csv file for the train images and the test images
feature_extraction(WORKING_DIR+"/metadataTrain.csv", WORKING_DIR+"/Train/Train/", "train_features.csv")
feature_extraction(WORKING_DIR+"/metadataTest.csv", WORKING_DIR+"/Test/Test/", "test_features.csv")

### Post-processing of extracted features

df_train = pd.read_csv("train_features.csv")
df_test = pd.read_csv("test_features.csv")

# Fill NaN for SEX class with the most frequent value
fill_SEX = df_train["SEX"].mode()[0]
df_train["SEX"] = df_train["SEX"].fillna(fill_SEX)
df_test["SEX"] = df_test["SEX"].fillna(fill_SEX)

# Fill NaN for AGE class with the most frequent value
fill_AGE = df_train["AGE"].mode()[0]
df_train["AGE"] = df_train["AGE"].fillna(fill_AGE)
df_test["AGE"] = df_test["AGE"].fillna(fill_AGE)

# Fill NaN for POSITION class with a new unknown class
df_train["POSITION"] = df_train["POSITION"].fillna("unknown")
df_test["POSITION"] = df_test["POSITION"].fillna("unknown")

# Fill NaN for numerical features with its mean value
fillna_features = ["f3", "f4", "f31", "f32", "f33"]
for f in fillna_features:
    fill = df_train[f].mean()
    df_train[f] = df_train[f].fillna(fill)
    df_test[f] = df_test[f].fillna(fill)

all_num_features = [elt for elt in df_train.columns if elt not in ["ID", "CLASS", "SEX", "POSITION"]]

# OneHot encoding of class SEX
df_train = df_train.replace({"SEX": {"male": 0, "female": 1}})
df_test = df_test.replace({"SEX": {"male": 0, "female": 1}})

# OneHot encoding of class POSITION
df_train = pd.get_dummies(df_train,prefix=[''], prefix_sep=[''], columns=['POSITION'], drop_first=True, dtype=int)
df_test = pd.get_dummies(df_test,prefix=[''], prefix_sep=[''], columns=['POSITION'], drop_first=True, dtype=int)

# Split train data in train and validation data
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=0)

# Normalization of train, val and test data
m = df_train[all_num_features].mean()
s = df_train[all_num_features].std()
df_train[all_num_features] = (df_train[all_num_features] - m)/s
df_val[all_num_features] = (df_val[all_num_features] - m)/s
df_test[all_num_features] = (df_test[all_num_features] - m)/s

# Deal with imbalanced data
# Reduce the number of samples of CLASS 2
newdf = df_train[df_train["CLASS"] == 2].sample(3500) 
df_train = pd.concat([df_train[df_train["CLASS"] != 2], newdf], ignore_index=True)

# Augment the number of samples of other CLASS
m = df_train["CLASS"].value_counts().values[0]
data_aug_rate = m / df_train["CLASS"].value_counts(sort=False).sort_index().values
for i in range(8):
    newdf = pd.DataFrame(np.repeat(df_train[df_train['CLASS'] == i+1].values, round(data_aug_rate[i]-1), axis=0), columns=df_train.columns)
    df_train=pd.concat([df_train, newdf])

# We export train, val and test csv with preprocessed features
df_train.to_csv("train_features_clean.csv", index=False)
df_val.to_csv("val_features_clean.csv", index=False)
df_test.to_csv("test_features_clean.csv", index=False)