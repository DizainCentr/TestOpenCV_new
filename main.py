# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pathlib import Path

import cv2


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
def print_coord_on_frame():
    print(1)
    p = Path(r'folder_for_frames')
    img = cv2.imread(str(r'folder_for_frames/0006.jpg'))
    cv2.line(img, (831, 454),(831, 453),(0, 0, 255), thickness=2)
    cv2.imwrite(r'folder_for_frames2/%04d.jpg' % 6, img)


if __name__ == '__main__':
    print_hi('PyCharm')
    print_coord_on_frame()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


