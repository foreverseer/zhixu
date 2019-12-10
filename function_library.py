"""
用来放一些跟绘图无关的函数，比如窗体置顶，文件操作等
目前我决定吧这个写成一个通用库
"""
import numpy as np
import re
import shutil
import os
import cv2
from PIL import Image
import imageio
import time
import win32gui
import win32con
from tqdm import tqdm


from numba import jit


class Bar:
    """
    进度条对象
    如果不显示进度条，则只会显示...Finish
    显示的话则会显示tqdm的默认进度条

    """

    def __init__(self, progress_bar=False):
        """
        使用前必须初始化

        :param progress_bar: 是否显示进度条，默认否
        """
        self.progress_bar = progress_bar

    def start(self, title=''):
        """
        展示标题

        :param title: 标题内容
        """
        if self.progress_bar:
            print('\033[0;31m' + title + '\033[0m', flush=True)
        else:
            print('\033[0;31m' + title + '\033[0m', flush=True, end='')
        time.sleep(0.001)

    def get_progress_bar(self, item_list, description=''):
        """
        将迭代器转换为tqdm进度条迭代器对象

        :param item_list: 迭代器
        :param description: 进度条标题，默认无
        :return:
        """
        if self.progress_bar:
            item_list = tqdm(item_list,ncols=80)
            item_list.set_description(description)
        return item_list

    def end(self):
        """
        过程结束，打印...Finish
        """

        if not self.progress_bar:
            print('\033[1;31m...Finish\033[0m')


def convert_images_to_gif(input_path, output_path, name, fps=50):
    """
    将一组图像转换为gif，注意输入的不是images，是路径，因为要转换为imageio的reader

    :param input_path: 输入路径
    :param output_path: 输出路径
    :param name: 输出文件名称
    :param fps: 帧率（似乎完全不影响输出速率）
    :return:
    """
    bar = Bar()
    bar.start('convert %s to %s.gif' % (input_path, os.path.join(output_path, name)))
    images_path = os.listdir(input_path)
    images_path = bar.get_progress_bar(images_path, 'load images from %s' % input_path)
    images = [imageio.imread(os.path.join(input_path, image_path)) for image_path in images_path]
    images = bar.get_progress_bar(images, 'convert to %s.gif' % os.path.join(output_path, name))
    imageio.mimsave(os.path.join(output_path, name) + '.gif', images, 'GIF', fps=fps)
    bar.end()


def set_window_forward(name):
    """
    将指定名称的window置顶

    :param name: 窗体名称
    :return:
    """
    hwnd_title = dict()

    def get_all_hwnd(hwnd_id):
        """
        获取所有的窗体的id，回调函数

        :param hwnd_id: 窗体id
        """
        if win32gui.IsWindow(hwnd_id) and win32gui.IsWindowEnabled(hwnd_id) and win32gui.IsWindowVisible(hwnd_id):
            hwnd_title.update({win32gui.GetWindowText(hwnd_id): hwnd_id})

    win32gui.EnumWindows(get_all_hwnd, 0)
    hwnd = hwnd_title[name]
    win32gui.SetForegroundWindow(hwnd)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 640, 480,
                          win32con.SWP_NOMOVE |
                          win32con.SWP_NOACTIVATE |
                          win32con.SWP_NOOWNERZORDER |
                          win32con.SWP_SHOWWINDOW)


@jit
def array_limit(array, up, down):
    """
    这个函数可以给一个数组设定上下限！！

    :param array: 输入数组
    :param up: 上限，高于上限的会等于上限
    :param down: 下限，低于下限的会等于下限
    :return: 返回标准化后的数组
    """
    u1 = array - np.array(up)
    u2 = (np.abs(u1) + u1) / 2
    array = array - u2
    d1 = np.array(down) - array
    d2 = (np.abs(d1) + d1) / 2
    array = array + d2
    return array


def path_cover(out_path):
    """
    对路径检查和覆写

    :param out_path: 输出路径
    :return:
    """
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)


def load_images(path, progress_bar=False):
    """
    加载一个队列的图像

    :param path: 图像路径
    :param progress_bar: 是否显示进度条，默认不显示
    :return: 一队列图像
    """
    '''
    加载一个队列的图像

    :param path: 图像路径
    :return: 一队列图像
    '''
    images_list = []
    dirs = os.listdir(path)
    if progress_bar:
        dirs = tqdm(dirs)
        dirs.set_description('loading_images')
    for file in dirs:
        images_list.append(Image.open(os.path.join(path, file)))
    return images_list


def save_images(images_list, types, path, progress_bar=True):
    """
    保存一个队列的图像

    :param images_list: 图像列表，image类型
    :param types: 保存图像的格式
    :param path: 保存图像的路径
    :param progress_bar: 是否显示进度条，默认不显示
    :return:
    """
    path_cover(path)
    if progress_bar:
        images_list = tqdm(images_list)
        images_list.set_description('save_images')
    for x, image in enumerate(images_list):
        image = image.convert('RGB')
        image.save('%s/%05d.%s' % (path, x, types))


def revise(revise_basic_list, convolution_kernel):
    """
    将将一个列表进行高斯平均（校正）

    :param revise_basic_list: 待校正列表
    :param convolution_kernel: 高斯平均的卷积核
    :return: 返回校正前的列表和矫正后的列表的比值列表
    """

    revise_list = revise_basic_list.copy()
    for x, i in enumerate(revise_basic_list):
        mins = x - convolution_kernel
        maxs = x + convolution_kernel
        if mins > 0:
            mins -= len(revise_basic_list)
            maxs -= len(revise_basic_list)
        revise_list[x] = np.mean(revise_basic_list[mins:] + revise_basic_list[:maxs])
    revise_list = np.array(revise_list) / np.array(revise_basic_list)
    return revise_list


def normalize_file_name(input_path, output_path, types='dicom'):
    """
    将文件目录规范化

    :param types: 保存文件的名称，用来替换原有名称
    :param input_path: 输入路径
    :param output_path: 输出路径，如果没有则自动创建 !!!如果是输出路径已经包含，则将原先目录覆盖
    :return:
    """
    bar = Bar()
    bar.start('normalize_file_name')
    re_numbers = re.compile(r'\d+')
    dirs = os.listdir(input_path)
    dirs = [file for file in dirs if len(re_numbers.findall(file))]
    dirs.sort(key=lambda file: int(re_numbers.findall(file)[0]))
    path_cover(output_path)
    for x, file in enumerate(bar.get_progress_bar(dirs)):
        os.popen('copy %s %s' % (os.path.join(input_path, file), os.path.join(output_path, '%05d.%s' % (x, types))))
    bar.end()


def convert_image_to_video(images, out_file, fps):
    """
    将输入的image阵列转换为视频，尽量输出为AVI

    :param images: 图像队列
    :param out_file: 输出视频路径和名称
    :param fps: 每秒帧数
    :return:
    """
    bar = Bar()
    bar.start('convert_image_to_video %s'%out_file)
    size = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*'h264')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）

    video = cv2.VideoWriter(out_file, fourcc, fps, size)
    for image in bar.get_progress_bar(images):
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()  # 释放
    bar.end()
