from PIL import Image, ImageEnhance
from pathlib import Path
import pytesseract
import argparse
import cv2
import os
from gooey import Gooey, GooeyParser

    
def main_write_process(gray, name, path):
    """
    Parameters:

    grays : list of preprocessed image object
    name : process name, as the placeholder for filename
    path : folder to save the file, chosen by user

    -----------------------------------------
    function writes the grayscale image to disk as a temporary file so we can
    apply OCR to it, then calls upon writefile function

    -----------------------------------------
    Returns : nothing

    """
    tpath="{}.txt".format(name)
    spath=os.path.join(path, tpath)
    filename = "{}.png".format(name)
    cv2.imwrite(filename, gray)
    text=ocr(filename)
    writefile(spath, text)
    

def ocr(filename):
    """
    Parameters:

    filename : image object to be loaded

    ------------------------------------
    function loads the image as a PIL/pillow image, apply OCR and then
    delete the temporary file

    ------------------------------------
    Returns : text file as OCR output

    """
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    return text


def writefile(target_dirs, text):
    """
    Parameters:

    target_dirs : path object (path of the file to be)

    --------------------------------------------------
    function checks if the file to be created has already existed
    if not, function tries to write the text file and flush it down to the hard
    drive

    --------------------------------------------------
    Returns : nothing if file object already existed
              else, writes a text file object and saved in hard drive

    """
    
    if os.path.exists(target_dirs):
        print("File already exists!")
        return

    f = open(target_dirs, 'w')
    try:
        f.write(text)
        f.flush()
        print("Done!")
    except Exception as e:
        print(e)
    finally:
        if f is not None:
            f.close()

@Gooey(program_name = "Optical Character Recognition")
def main():
    # construct the argument parse and parse the arguments
    ap = GooeyParser()
    group=ap.add_argument_group("Pengaturan", "Tentukan apa yang diproses dan bagaimana prosesnya")
    group.add_argument("-i", "--image", metavar="Gambar",
            help="Pilih gambar yang mau diproses", widget="FileChooser")
    group.add_argument("-p", "--preprocess", metavar="Mode Proses", type=str, default="thresh",
            help="Pilih jenis pemrosesan (jika tidak ada, default adalah threshold yaitu mempertajam gambar)", choices=["Threshold", "Blur"], widget="Listbox", nargs='*')
    group.add_argument("-d", "--dirofimage", metavar="Folder",
            help="Pilih folder lokasi gambar yang akan diproses", widget="DirChooser")
    group.add_argument("-f", "--targetfolder", metavar="Simpan", help="Pilih lokasi penyimpanan", widget="DirChooser")
    args = ap.parse_args()


    grays=[] #list to contain image file
    if args.dirofimage:
            #load files from folder
            strPath=args.dirofimage #store input path in variable

            for roots, dirs, files in os.walk(strPath):
                    pass

            iterator_number=0
            for file in files:
                    #load example one by one and convert to grayscale
                    abs_path=os.path.join(roots, file)
                    img=cv2.imread(abs_path)
                    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    iterator_number+=1
                    name=os.getpid()+iterator_number
                    main_write_process(gray, name)
                    
                    
    if args.image:
            # load the example image and convert it to grayscale
            image = cv2.imread(args.image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grays.append(gray)
            iterator_number=0
            iterator_number+=1
            name=os.getpid()+iterator_number


    if args.preprocess == "Threshold":
            # check to see if we should apply thresholding to preprocess the image
            for gray in grays:
                    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,11,2)


    elif args.preprocess == "Blur":
            # make a check to see if median blurring should be done to remove noise
            for gray in grays:
                    gray = cv2.medianBlur(gray, 3)
    
    path=args.targetfolder
    main_write_process(gray, name, path) #do the whole process OCR->writefile

main()
