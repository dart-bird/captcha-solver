import cv2 
import numpy as np
from numpy import asarray
from PIL import Image
import pytesseract

def load_image( infilename, mode ) :
    img = Image.open( infilename)
    img = img.convert(mode)
    img.load()
    data = np.asarray( img, dtype="uint8" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "RGB" )
    img.save( outfilename )

def show_image( npdata ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "RGB" )
    img.show()

def crop_from_img(img: np.array):
    result = np.where(img != 255)
    mid_cord = result[0][int(len(result[0])//2)]
    return img[abs(mid_cord-15):mid_cord+15, 0:100]

def img_Contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def solve_cap(filename):
    path = 'images/{}.png'.format(filename)

    np_img = load_image(path, "RGB")

    r_data, g_data, b_data = np_img[:,:,0], np_img[:,:,1], np_img[:,:,2]
    r_modified_data = []
    g_modified_data = []
    b_modified_data = []

    for np_ in r_data:
        line_pixel = np_[-1]
        r_modified_data.append(np.where((line_pixel == np_) | (0 == np_) , 255, np_))
    np_img[:,:,0] = r_modified_data
    g_modified_data = []

    for np_ in g_data:
        line_pixel = np_[-1]
        g_modified_data.append(np.where((line_pixel == np_) | (0 == np_) , 255, np_))
    np_img[:,:,1] = g_modified_data
    b_modified_data = []

    for np_ in b_data:
        line_pixel = np_[-1]
        b_modified_data.append(np.where((line_pixel == np_) | (0 == np_) , 255, np_))
    np_img[:,:,2] = b_modified_data

    r_data, g_data, b_data = np.transpose(np_img[:,:,0]), np.transpose(np_img[:,:,1]), np.transpose(np_img[:,:,2])

    r_modified_data = []
    g_modified_data = []
    b_modified_data = []

    for np_ in r_data:
        line_pixel = np_[-1]
        r_modified_data.append(np.where((line_pixel == np_) | (0 == np_) , 255, np_))
    np_img[:,:,0] = np.transpose(r_modified_data)
    g_modified_data = []

    for np_ in g_data:
        line_pixel = np_[-1]
        g_modified_data.append(np.where((line_pixel == np_) | (0 == np_) , 255, np_))
    np_img[:,:,1] = np.transpose(g_modified_data)
    b_modified_data = []

    for np_ in b_data:
        line_pixel = np_[-1]
        b_modified_data.append(np.where((line_pixel == np_) | (0 == np_) , 255, np_))
    np_img[:,:,2] = np.transpose(b_modified_data)


    im = Image.fromarray(np_img)
    im = im.convert('L')
    gray_img = np.asarray( im, dtype="uint8" )
    gray_modified_img = []
    for img_data in gray_img:
        gray_modified_img.append(np.where(255 != img_data, 0, img_data))

    modified_img = np.array(gray_modified_img)
    im2 = Image.fromarray(crop_from_img(modified_img))
    txt = pytesseract.image_to_string(im2, config='--psm 7 --oem 1').rstrip().replace(' ','')
    txt = ''.join(char for char in txt if char.isalnum())
    print(filename, txt)


if __name__ == "__main__":
    for filename in range(100):
        solve_cap(filename)