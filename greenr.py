import cv2
import numpy as np
from tkinter import Tk, filedialog, Button

def select_file():
    file_path = filedialog.askopenfilename()
    print(f'Selected file: {file_path}')
    return file_path

def read_image(file_path):
    img = cv2.imread(file_path)
    return img
def select_green_screen_image():
    global green_screen_img
    green_screen_img = read_image(select_file())

def select_replacement_image():
    global replacement_img
    replacement_img = read_image(select_file())

root = Tk()
root.geometry("300x100")

btn1 = Button(root, text="选择绿幕图片", command=select_green_screen_image)
btn1.pack()

btn2 = Button(root, text="选择替换图片", command=select_replacement_image)
btn2.pack()

root.mainloop()

img=green_screen_img
replace = replacement_img
# 定义一个函数，用于鼠标回调，保存点击的角点
points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f'Point collected: ({x}, {y})')

# 获取屏幕分辨率
screen_width = 1920  # 示例值，根据实际情况替换
screen_height = 1080  # 示例值，根据实际情况替换

# 计算缩放比例，确保图像不会超出屏幕
scale_width = screen_width * 0.9 / img.shape[1]
scale_height = screen_height * 0.9 / img.shape[0]
scale = min(scale_width, scale_height)

# 缩放图像
if scale < 1:
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# 显示图像并等待用户点击四个角点
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Image', mouse_callback)
while len(points) < 4:
    cv2.imshow('Image', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()

# 顺序保存为 top-left, top-right, bottom-right, bottom-left
top_left, top_right, bottom_right, bottom_left = points

# 计算新图像的大小
width_A = np.sqrt((bottom_right[0] - bottom_left[0]) ** 2 + (bottom_right[1] - bottom_left[1]) ** 2)
width_B = np.sqrt((top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_left[1]) ** 2)
max_width = replace.shape[1]

height_A = np.sqrt((top_right[0] - bottom_right[0]) ** 2 + (top_right[1] - bottom_right[1]) ** 2)
height_B = np.sqrt((top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2)
max_height = replace.shape[0]

# 透视变换的源点和目标点
dst_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
src_pts = np.float32([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]])

# 调整背景图片的大小
replace_resized = cv2.resize(replace, (max_width, max_height))

# 获取透视变换矩阵并应用它
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped_replace = cv2.warpPerspective(replace_resized, matrix, (img.shape[1], img.shape[0]))

# 创建一个掩码，用于将绿幕区域与变换后的背景合并
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.fillConvexPoly(mask, np.int32(dst_pts), (255, 255, 255), cv2.LINE_AA)

# 创建一个反向掩码，用于从原图中提取非绿幕区域
mask_inv = cv2.bitwise_not(mask)

# 从原图中提取绿幕区域
img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
warped_replace_fg = cv2.bitwise_and(warped_replace, warped_replace, mask=mask)

# 合成最终图像
final_output = cv2.add(img_bg, warped_replace_fg)

cv2.imwrite('final_image.jpg', final_output)
