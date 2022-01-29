from __future__ import division
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


from models.experimental import attempt_load
from utils import utils, general
from utils.general import check_img_size, scale_coords
from utils.utils import *
from utils.datasets import *

import os
import time

import argparse

from PIL import Image

import torch
from material_detect_img import Ui_material_detect_img
import cv2
import numpy


class mywindow(QtWidgets.QMainWindow, Ui_material_detect_img):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--imgs_path", type=str,
                                 default="D:/code(last)/yolo/yolo5/yolov5-ultralytics/data/custom/valid",
                                 help="读取图片文件夹路径，多张图片")  # TODO 改
        self.parser.add_argument("--weights_path", type=str,
                                 default="checkpoints/2021-07-16_22-01-09_yolov5m/weights/best.pt",
                                 help="模型 或者 weights/yolov5x.pt 路径")  # TODO 改
        self.parser.add_argument('--data', type=str, default='data/voc.yaml', help='数据.yaml 路径')
        self.parser.add_argument("--conf_thres", type=float, default=0.2, help="对象置信阈值，识别率")
        self.parser.add_argument("--nms_thres", type=float, default=0.3, help="IOU限制 对于非最大值抑制")
        self.parser.add_argument("--img_size", type=int, default=320, help="每个图像维度的大小")
        self.parser.add_argument("--isGpu", default=False, action='store_true', help="是否开启GPU，符合条件才能开启")
        self.parser.add_argument('--isShowName', default=True, action='store_true', help="显示类名")
        self.parser.add_argument('--isShowNumber', default=False, action='store_true', help="显示序号")
        self.parser.add_argument('--isShowConf', default=True, action='store_true', help="显示 conf_thres")
        self.parser.add_argument("--line_thickness", type=int, default=2, help="标签框的大小，宽度")
        self.parser.add_argument('--augment', action='store_true', default=False, help='增广推理')
        self.parser.add_argument('--agnostic_nms', action='store_true', default=False, help='一种计算方法')
        self.parser.add_argument('--isVideoFixedSize', action='store_true', default=True, help='是否统一固定图片大小')
        self.parser.add_argument("--img_show_w", type=int, default=256, help="图片显示宽度")
        self.parser.add_argument("--img_show_h", type=int, default=256, help="图片显示高度")
        self.parser.add_argument("--ask_iou", default=False, help="是否计算iou")
        self.output_img_s = []
        self.iou = []
        self.iou_1 = []
        self.iou_average = []
        self.opt = self.parser.parse_args()

        self.b_cuda = torch.cuda.is_available()
        if not self.opt.isGpu and self.b_cuda == True:
            self.b_cuda = False
        self.opt.isGpu = self.b_cuda
        self.time11 = time.time()
        self.weightslist = ['yolov5l.pt', 'yolov5l6.pt', 'yolov5m.pt', 'yolov5m6.pt', 'yolov5s.pt', 'yolov5s6.pt',
                            'yolov5x.pt',
                            'yolov5x6.pt']
        self.timer_camera = QTimer()
        self.opt = self.parser.parse_args()
        self.myImagesToSingle = ImagesToSingle()  # 处理方法封装
        self.str_root = ""
        self.start = False
        self.single_picture_detect_state = False
        self.start_show=False
        self.btn_start.clicked.connect(self.begin_start)
        self.btn_choose_single_picture.clicked.connect(self.single_picture_detect)
        self.btn_material_detect.clicked.connect(self.material_detect_start)
        self.btn_forward.clicked.connect(self.show_img_forward)
        self.btn_backward.clicked.connect(self.show_img_backward)
        self.btn_cancel.clicked.connect(self.clear)
        self.directory_s = []
        self.directory = []
        self.i = 0
        self.start_num=0

    def clear(self):
        self.directory_s = []
        self.directory = []
        self.i = 0
        self.str_root = ""
        self.start = False
        self.single_picture_detect_state = False
        self.start_show = False
        self.output_img_s = []
        self.iou = []
        self.iou_1 = []
        self.iou_average = []
        self.label_big_source_picture.setPixmap(QPixmap(""))
        self.label_big_material_detect_picture.setPixmap(QPixmap(""))
        self.txt_file_root.clear()
        self.txt_msg.clear()
        self.label_already_shown_num.clear()
        self.label_show_fps_assume.clear()
        self.label_iou_average.clear()
        self.label_iou_all_average.clear()
        self.start_num = 0

    def closeEvent(self, event):
        event.accept()
        os._exit(0)

    def getModel(self, opt):
        b_cuda = torch.cuda.is_available()
        if not opt.isGpu and b_cuda:
            b_cuda = False
        device = torch.device("cuda" if b_cuda else "cpu")
        # Set up model
        model = attempt_load(opt.weights_path, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(opt.img_size, s=stride)  # check img_size
        half = b_cuda  # half precision only supported on CUDA
        if half:
            model.half()  # to FP16
        if b_cuda:
            model.cuda()
        if b_cuda:
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        model.eval()  # 在评估模式下设置,不随机舍弃神经元
        classes = load_classes(opt.data)  # Extracts class labels from file
        colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]
        # 指定颜色，红、蓝、绿
        if len(colors) <= 3 and len(colors) > 0:
            # BGR 0,0,255 | 255,0,0  |  0,255,0
            # RGB 255,0,0 | 0,0,255  |  0,255,0
            colors[0] = [0, 0, 255]  # 红色
            if len(colors) >= 2:
                colors[1] = [255, 0, 0]  # 蓝色
            if len(colors) >= 3:
                colors[2] = [0, 255, 0]  # 绿色
        return model, classes, colors, device, stride, imgsz

    def begin_start(self):
        if self.start_num==0 :
            self.start = True
            self.single_picture_detect_state = False
            self.start_num=1
        else:
            QMessageBox.information(self, '提示', '已启动，若想重新启动请先退出!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    def single_picture_detect(self):
        try:
            directory_temp=[]
            self.single_picture_detect_state = True
            if self.start:
                directory_temp, filetype = QFileDialog.getOpenFileNames(self, "选取文件")  # 起始路径
                if len(directory_temp)>0:
                    for i in range(len(directory_temp)):
                        directory1 = directory_temp[i]
                        str_root = str(directory1)
                        f_PNG = str_root.rfind('.PNG')
                        f_png = str_root.rfind('.png')
                        f_jpg = str_root.rfind('.jpg')
                        f_jpeg = str_root.rfind('.jpeg')
                        if f_jpg == -1 and f_jpeg == -1 and f_png == -1 and f_PNG == -1:
                            self.txt_file_root.append(str_root)
                            QMessageBox.information(self, '提示', '上传错误，请重新拍照或者上传正确格式的照片!', QMessageBox.Ok | QMessageBox.Close,
                                                QMessageBox.Close)
                        else:
                            self.txt_file_root.append(str_root)
                            self.directory_s.append(directory_temp[i])
                    QMessageBox.information(self, '提示', '上传成功!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
                else:
                    QMessageBox.information(self, '提示', '未输入图片!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, '提示', '请先启动!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        except:
            QMessageBox.information(self, '提示', '程序出错!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    def material_detect_start(self):
        try:
            if self.start:
                if self.single_picture_detect_state:
                    opt = self.opt
                    model, classes, colors, device, stride, imgsz = self.getModel(opt)
                    listCentre = []  # 记录中心点 listCentre = [(128,128)]
                    isLine = False  # 是否绘制轨迹点

                    intVideoFixedW = opt.img_show_w  # 宽度
                    intVideoFixedH = opt.img_show_h  # 高度
                    dFixedProportion = intVideoFixedW / intVideoFixedH  # 宽高比
                    print("图片显示宽高比:%d"%dFixedProportion)

                    for i in range(len(self.directory_s)):
                        self.txt_msg.append("第%d张:"%(i+1))
                        source = self.directory_s[i]
                        path, name = os.path.split(source)
                        imagePIL = Image.open(source)
                        imagePIL = imagePIL.convert("RGB")
                        imageCV = cv2.cvtColor(numpy.asarray(imagePIL), cv2.COLOR_RGB2BGR)

                        shape = imageCV.shape[:2]  # 图片大小

                        # 图片转换    ->320
                        input_torch = self.myImagesToSingle.LoadImagesToSingleAndCV2(imageCV, stride=stride,
                                                                                     imgsz=imgsz,
                                                                                     device=device)
                        # 绘制中心点
                        if isLine and len(listCentre) > 0:
                            for item in listCentre:
                                self.myImagesToSingle.plot_one_circle(item, imageCV, point_color=[255, 0, 0],
                                                                      point_size=2)
                        # 调用模型识别图片
                        with torch.no_grad():
                            pred = model(input_torch, augment=opt.augment)[0]
                            detections = general.non_max_suppression(pred, opt.conf_thres, opt.nms_thres,
                                                                     classes=None,
                                                                     agnostic=opt.agnostic_nms)    # 非极大值抑制 阈值

                        time1 = time.time()                # strftime() 函数接收以时间元组，并返回以可读字符串表示的当地时间


                        # 绘制边界框和检测标签
                        classCount = 0
                        if detections is not None:
                            detectionsLen = len(detections)
                            if detectionsLen > 0:
                                detections2 = detections[0]
                                detectionsLen2 = len(detections2)
                                if detectionsLen2 > 0:
                                    # 将框重缩放到原始图像
                                    detections2[:, :4] = scale_coords(input_torch.shape[2:], detections2[:, :4],
                                                                      shape).round()
                                    detections3 = detections2
                                    detectionsLen3 = len(detections2)
                                    reserve = 5
                                    imgW = shape[1] - reserve
                                    imgH = shape[0] - reserve

                                    if detections3 is not None and detectionsLen3 > 0:
                                        for x1, y1, x2, y2, cls_conf, cls_pred in detections3:
                                            if cls_conf < 0.3:             # conf过小忽略
                                                continue

                                            x1 = x1.item()  # 取出张量具体位置的元素元素值，并且返回的是该位置元素值的高精度值，保持原元素类型不变；必须指定位置
                                            y1 = y1.item()
                                            x2 = x2.item()
                                            y2 = y2.item()
                                            if x1 < reserve:
                                                x1 = reserve
                                            if y1 < reserve:
                                                y1 = reserve
                                            if x2 < reserve:
                                                x2 = reserve
                                            if y2 < reserve:
                                                y2 = reserve
                                            if x2 > imgW:
                                                x2 = imgW
                                            if y2 > imgH:
                                                y2 = imgH
                                            box_w = x2 - x1
                                            box_h = y2 - y1
                                            classesName2 = classes[int(cls_pred)]
                                            box1 = [int(x1), int(y1), int(box_w), int(box_h)]
                                            boxText = 'x:' + str(box1[0]) + ',y:' + str(
                                                box1[1]) + ',w:' + str(
                                                box1[2]) + ',h:' + str(box1[3])
                                            box = [x1, y1, x2, y2]  # 矩形

                                            # 判断是否计算iou
                                            if path == self.opt.imgs_path:
                                                self.opt.ask_iou = True
                                            else:
                                                self.opt.ask_iou=False

                                            # 读入当前图片labelme标注
                                            if self.opt.ask_iou==True:
                                                iou=self.if_compute_iou(name,box,int(cls_pred))

                                                self.iou.append(iou)
                                                self.iou_1.append(iou)
                                                classCount += 1
                                                msg = ("(%d)%s,conf:%.3f,iou:%.3f,%s" % (
                                                    classCount,classesName2, cls_conf.item(), iou, boxText))
                                            else:
                                                classCount += 1
                                                msg = ("(%d)%s,conf:%.3f,%s" % (
                                                    classCount, classesName2, cls_conf.item(),  boxText))
                                            self.txt_msg.append(msg)
                                            classesName = ''
                                            if opt.isShowName:
                                                classesName = classesName2
                                            if opt.isShowNumber:
                                                if classesName != '':
                                                    classesName = classesName + str(classCount)
                                                else:
                                                    classesName = str(classCount)
                                            if opt.isShowConf:
                                                if classesName != '':
                                                    classesName = classesName + str(round(cls_conf.item(), 3))
                                                else:
                                                    classesName = str(round(cls_conf.item(), 3))

                                            labColor = colors[int(cls_pred)]  # 标签 颜色
                                            boxColor = colors[int(cls_pred)]  # 红色
                                            self.myImagesToSingle.plot_one_box(box, imageCV, label=classesName,
                                                                               labColor=labColor,
                                                                               boxColor=boxColor,
                                                                               line_thickness=opt.line_thickness)  # 画框 和 画标签
                                        if i<len(self.directory_s)-1:
                                            self.txt_msg.append("-------------------------------------------------")

                        self.output_img_s.append(imageCV)
                        if self.opt.ask_iou == True:
                            iou_average_1 = round(sum(self.iou)/len(self.iou),4)
                            self.iou.clear()
                            self.iou_average.append(iou_average_1)
                        time2 = time.time()
                        strTime = round((time2 - time1), 4)
                        if time2 - time1 != 0:
                            time_fps = round(1 / (time2 - time1), 4)
                            msg = "耗时:" + str(strTime) + ',FPS:' + str(time_fps)
                            self.label_show_fps_assume.setText(msg)
                    self.single_picture_detect_state = False
                    self.start = False
                    self.start_show=True
                    self.show_im()
                    QMessageBox.information(self, '提示', '图片全部检测完毕', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
                else:
                    QMessageBox.information(self, '提示', '材料有错或者未采集到，请重新准备好识别材料!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, '提示', '请先启动!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        except:
            QMessageBox.information(self, '提示', '程序出错，请重新检查!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    def show_im(self):
        img = cv2.imread(self.directory_s[self.i])
        img = cv2.resize(img, (256, 256))
        frame1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width, bytesPerComponent = frame1.shape
        bytesPerLine = bytesPerComponent * width
        q_source_image = QImage(frame1.data, width, height, bytesPerLine,
                                QImage.Format_RGB888).scaled(self.label_big_source_picture.width(),
                                                             self.label_big_source_picture.height())
        self.label_big_source_picture.setPixmap(QPixmap.fromImage(q_source_image))
        #self.source_picture.close()
        img_output = cv2.resize(self.output_img_s[self.i], (256, 256))
        frame2 = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
        height, width, bytesPerComponent = frame2.shape
        bytesPerLine = bytesPerComponent * width
        q_material_detect_image = QImage(frame2.data, width, height, bytesPerLine,
                                         QImage.Format_RGB888).scaled(
            self.label_big_material_detect_picture.width(),
            self.label_big_material_detect_picture.height())
        self.label_big_material_detect_picture.setPixmap(QPixmap.fromImage(q_material_detect_image))
        #self.detect_material_picture.close()
        if self.opt.ask_iou == True:
            self.label_iou_average.setText("当前图片iou均值:" + str(self.iou_average[self.i]))
            iou_all_average = round(sum(self.iou_1) / len(self.iou_1), 4)
            self.label_iou_all_average.setText("本轮所有图片的iou均值:" + str(iou_all_average))
        else:
            self.label_iou_average.setText("")
            self.label_iou_all_average.setText("")
        self.label_already_shown_num.setText("当前第%d张/共%d张" % (self.i + 1, len(self.directory_s)))

    def show_img_forward(self):
        if self.start_show:
            self.i=self.i-1
            if self.i == -1:
                self.i = 0
                QMessageBox.information(self, '提示', '这是第一张', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            else:
                self.show_im()

    def show_img_backward(self):
        if self.start_show:
            self.i = self.i + 1
            if self.i == len(self.directory_s):
                self.i = len(self.directory_s)-1
                QMessageBox.information(self, '提示', '已是最后一张', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            else:
                self.show_im()

    def if_compute_iou(self,name,box,class_num):
        with open(r"data\custom\all.txt", "r") as f1:
            list_name = f1.readlines()
            for j in range(len(list_name)):
                    list_name[j] = str(list_name[j]).replace("\n", "")
            for k in list_name:
                if k == str(name).replace(".jpg", ".txt"):
                    path = "data\custom\labels\\" + k
                    with open(path, "r") as f:
                        all_box = f.read().split("\n")[:-1]                        # 读取提前做好的label值
                        iou_best=0
                        for box_num in range(len(all_box)):
                            all_box_each = all_box[box_num].split(" ")

                            if int(all_box_each[0])==class_num:         # 反归一化
                                temp_x1=int((float(all_box_each[1])-0.5*float(all_box_each[3]))*256)
                                temp_y1 = int((float(all_box_each[2])-0.5*float(all_box_each[4]))*256)
                                temp_x2 = int((float(all_box_each[1])+0.5*float(all_box_each[3]))*256)
                                temp_y2 = int((float(all_box_each[2])+0.5*float(all_box_each[4]))*256)
                                temp_box=[temp_x1,temp_y1,temp_x2,temp_y2]
                                if bbox_iou(box,temp_box)>=iou_best:
                                    iou_best=bbox_iou(box,temp_box)
        return iou_best



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = mywindow()
    ui.setStyleSheet("#material_detect_img{border-image:url(beijing/blue.jpg);}")
    ui.show()
    sys.exit(app.exec_())
