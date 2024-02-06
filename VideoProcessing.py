import math
import os
from pathlib import Path

import cv2 as cv

import numpy as np

def increase_brightness(img, value=100):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img


def find_points1(thresh, center=(831, 454)):
    find_points = []
    for k in range(72):
        print('k: ', k)
        max_distance = 0
        max_x = 0
        max_y = 0
        for i in range(len(thresh)):
            for j in range(len(thresh[i])):
                # if i>=0 and j>=0:
                if thresh[i][j] >= 240:
                    distance_x = (j - center[0])
                    distance_y = (i - center[1])
                    distance = math.sqrt(distance_x * distance_x + distance_y * distance_y)
                    alfa = math.atan2(distance_y, float(distance_x))

                    if (k*5 - 3)/180*math.pi <= alfa < (k*5 + 3)/180*math.pi:

                        if distance > max_distance:
                            print('alfa: ', alfa, 'max_distance: ', distance, 'max_x: ', j, 'max_y: ',i)
                            max_distance = distance
                            max_x = j
                            max_y = i
        if max_distance == 0:
            pass
        else:
            print('find_points:', max_x, max_y)
            find_points.append((max_x, max_y))
    print('find_points: ', find_points)
    return find_points

def get_frames():
    cap = cv.VideoCapture("Project 3.avi")  # вывод кадров из видео файла
    ret, frame1 = cap.read()
    numb=0
    while cap.isOpened():
        numb+=1
        try:
            cv.imshow("frame1", frame1)
        except:
            break
        cv.imwrite(r'folder_for_frames/%04d.jpg' % numb, frame1)
        if cv.waitKey(40) == 27:
            break
        try:
            for i in  range(5):
                ret, frame1 = cap.read()
        except:
            break

def moving_traker5():
    import cv2  # импорт модуля cv2

    cap = cv2.VideoCapture("Project 3.avi")  # вывод кадров из видео файла
    center = (831, 454)
    # cap = cv2.VideoCapture(0)  # видео поток с веб камеры

    cap.set(3, 1280)  # установка размера окна
    cap.set(4, 700)

    ret, frame0 = cap.read()
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    kernel = np.ones((5, 5), np.uint8)
    numb = 0
    try:
        os.mkdir(r'folder_for_frames')
    except FileExistsError:
        pass
    while cap.isOpened():  # метод isOpened() выводит статус видеопотока
        for i in range(5):
            try:
                _, frame2 = cap.read()
            except:
                break
        numb += 1
        print(numb)
        try:
            diff = cv2.absdiff(frame1,
                               frame2)  # нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них, т.е. с этого момента наша программа реагирует на любое движение.
        except:
            print("не удалось выполнить вычитание кадров")
            break
        img1 = increase_brightness(diff, 10)
        img = cv.dilate(img1, kernel, iterations=1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        сontours, _ = cv2.findContours(thresh, cv.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)  # нахождение массива контурных точек

        tmp_frame = img

        cv2.drawContours(tmp_frame, сontours, -1, (255, 255, 0), 2) #также можно было просто нарисовать контур объекта

        # dilated = cv2.dilate(thresh, None, iterations=3)

        find_points = find_points1(thresh)
        for i in range(len(find_points)-1):


            cv2.line(tmp_frame, (find_points[i][0], find_points[i][1]), (find_points[i+1][0], find_points[i+1][1]), (0, 255, 0), thickness=2)
        cv2.circle(tmp_frame, center, 5,(0,0,255))
        cv2.line(tmp_frame, (find_points[-1][0], find_points[-1][1]), (find_points[0][0], find_points[0][1]),
                 (0, 255, 0), thickness=2)
        cv2.imshow("frame1", tmp_frame)

        cv2.imwrite(r'folder_for_frames/%04d.jpg' % numb, tmp_frame)
        if cv2.waitKey(40) == 27:
            break
        frame1 = frame2  #


    cap.release()
    cv2.destroyAllWindows()
    p = Path(r'folder_for_frames')
    imgs = [cv2.imread(str(f)) for f in p.glob('*.jpg')]
    writer = cv2.VideoWriter(
        r'../new_video.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        25.0,  # 25
        (imgs[0].shape[1], imgs[0].shape[0]),
        isColor=len(imgs[0].shape) > 2)
    for frame in imgs:
        writer.write(frame)
    writer.release()


def moving_traker4():
        # import OpenCV and pyplot
        import cv2 as cv
        from matplotlib import pyplot as plt
        p = Path(r'folder_for_frames')
        # imgs = [cv.imread(str(f)) for f in p.glob('*.jpg')]
        tmp = [str(f) for f in p.glob('*.jpg')]
        # read left and right images
        for i in range(len(tmp)-1):
            imgR = cv.imread(tmp[i], 0)
            imgL = cv.imread(tmp[i+1], 0)
            # creates StereoBm object
            stereo = cv.StereoBM_create(numDisparities=16,
                                        blockSize=15)

            # computes disparity
            disparity = stereo.compute(imgL, imgR)

            # displays image as grayscale and plotted
            cv.imshow("frame1", disparity)

            cv.imwrite(r'folder_for_frames1/%04d.jpg' % i, disparity)
            # plt.imshow(disparity, 'gray')
            # plt.show()





def moving_traker6():
    import cv2  # импорт модуля cv2

    cap = cv2.VideoCapture("Project 3.avi")  # вывод кадров из видео файла

    # cap = cv2.VideoCapture(0)  # видео поток с веб камеры

    cap.set(3, 1280)  # установка размера окна
    cap.set(4, 700)

    ret, frame0 = cap.read()
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    kernel = np.ones((5, 5), np.uint8)
    numb = 0
    try:
        os.mkdir(r'folder_for_frames')
    except FileExistsError:
        pass
    while cap.isOpened():  # метод isOpened() выводит статус видеопотока
        for i in range(5):
            try:
                _, frame2 = cap.read()
            except:
                break
        numb += 1
        print(numb)
        try:
            diff = cv2.absdiff(frame1,
                               frame2)  # нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них, т.е. с этого момента наша программа реагирует на любое движение.
        except:
            print("не удалось выполнить вычитание кадров")
            break
        img1 = increase_brightness(diff, 10)
        img = cv.dilate(img1, kernel, iterations=1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        сontours, _ = cv2.findContours(thresh, cv.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)  # нахождение массива контурных точек
        tmp_frame = img
        for j in range(360):
            max_distance = 0
            for item in thresh:
                pass


        # cv2.drawContours(tmp_frame, сontours, -1, (0, 255, 0), 2) #также можно было просто нарисовать контур объекта

        # dilated = cv2.dilate(thresh, None, iterations=3)

        cv2.imshow("frame1", tmp_frame)

        cv2.imwrite(r'folder_for_frames/%04d.jpg' % numb, tmp_frame)
        if cv2.waitKey(40) == 27:
            break
        frame1 = frame2  #


    cap.release()
    cv2.destroyAllWindows()
    p = Path(r'folder_for_frames')
    imgs = [cv2.imread(str(f)) for f in p.glob('*.jpg')]
    writer = cv2.VideoWriter(
        r'../new_video.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        25.0,  # 25
        (imgs[0].shape[1], imgs[0].shape[0]),
        isColor=len(imgs[0].shape) > 2)
    for frame in imgs:
        writer.write(frame)
    writer.release()

if __name__ == '__main__':
    # get_frames()
    moving_traker6()