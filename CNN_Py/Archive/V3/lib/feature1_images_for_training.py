
import cv2
import os 
import glob


def get_paths(folder_name_string):
    im_dir = os.path.join(os.getcwd(), folder_name_string)
    paths = glob.glob(os.path.join(im_dir, "*.jpg"))
    paths = list(paths)
    return paths


def cropped_imgs(hr_paths):
    
    image_size = 32
    color_dim = 3
    step = 21
    padding = 6

    HR_sequence = []
    
    for i in range(len(hr_paths)):

            HR=cv2.imread(hr_paths[i])
            
            if len(HR.shape) == 3: 
                h, w, c = HR.shape
            else:
                h, w = HR.shape 
     
            nx, ny = 0, 0
            for x in range(0, h - image_size + 1, step):
                nx += 1; ny = 0
                for y in range(0, w - image_size + 1, step):
                    ny += 1
    
                    HR_cropped = HR[x + padding: x + padding + image_size, y + padding: y + padding + image_size] 
                    HR_cropped =  HR_cropped / 255.0
                    HR_sequence.append(HR_cropped)

    return HR_sequence



