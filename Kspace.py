from PyQt5 import QtWidgets ,QtCore,QtGui
from PyQt5.QtWidgets import (QMainWindow, QTextEdit,QAction, QFileDialog, QApplication,QSlider,QTabWidget,QWidget)
from PyQt5.QtWidgets import QLabel, QMessageBox
from PyQt5.QtGui import QIcon, QPixmap
import sys
import numpy as np
from scipy import fftpack
import pyqtgraph as pg
import matplotlib.pyplot as plt
import cv2 as cv
from  mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from GUI import Ui_MainWindow

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionOpen_Image.triggered.connect(self.OpenImage)
        self.ui.Combo1.currentTextChanged.connect(self.ChooseOperation)
        self.ui.comboBox.currentTextChanged.connect(self.Gradients)
        self.ui.Combo1.setCurrentIndex(0)
        self.widgets = [self.ui.InputImage1,self.ui.FourierInput1]
        for i in range(len(self.widgets)):
            self.widgets[i].ui.histogram.hide()
            self.widgets[i].ui.roiBtn.hide()
            self.widgets[i].ui.menuBtn.hide()
            self.widgets[i].ui.roiPlot.hide()
        self.pen1 = pg.mkPen(color=(255, 0, 0))
        self.pen2 = pg.mkPen(color=(0, 0, 255))
        self.ui.pushButton.clicked.connect(self.ShowGraph)

    def ShowGraph (self):
        np.random.seed(42)
        rand_Bz = np.random.randint(-3,4,20)
        
        const_Bx = [0]*20
        const_By = [0]*20
        const_Bz = [2]*20
        nonUn_Bz = rand_Bz + const_Bz

        self.ui.withx.clear()
        self.ui.withy.clear()
        self.ui.withz.clear()

        self.ui.withz.plotItem.plot(nonUn_Bz,pen=self.pen1)
        self.ui.withz.setLabel('left', 'B', units='Tesla')
        self.ui.withz.setLabel('bottom', 'Z-axis')
        self.ui.withx.plotItem.plot(const_Bx,pen=self.pen2)
        self.ui.withx.setLabel('left', 'B', units='Tesla')
        self.ui.withx.setLabel('bottom', 'X-axis')
        self.ui.withy.plotItem.plot(const_By,pen=self.pen2)
        self.ui.withy.setLabel('left', 'B', units='Tesla')
        self.ui.withy.setLabel('bottom', 'Y-axis')


    def Gradients (self):
        var = np.arange(-1,1,0.1)
        if (str(self.ui.comboBox.currentText())) == 'Gradient Effect':
            print('Choose prope Gradient Coil')
        elif (str(self.ui.comboBox.currentText())) == 'Slice Selection':
            const_Bx = [0]*20
            const_BxG = const_Bx + var
            self.ui.withx.clear()
            self.ui.withx.plotItem.plot(const_BxG,pen=self.pen2)
        elif (str(self.ui.comboBox.currentText())) == 'Phase Encoding':
            const_By = [0]*20
            const_ByG = const_By + var
            self.ui.withy.clear()
            self.ui.withy.plotItem.plot(const_ByG,pen=self.pen2)
        elif (str(self.ui.comboBox.currentText())) == 'Frequency Encoding':
            const_Bz = [2]*20
            rand_Bz = np.random.randint(-3,4,20)
            nonUn_Bz = rand_Bz + const_Bz
            const_BzG = nonUn_Bz + var
            self.ui.withz.clear()
            self.ui.withz.plotItem.plot(const_BzG,pen=self.pen1)


    def OpenImage(self):
        self.filePath = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', 'F:\25-9\Downloads\talta tebya\Second Semester\DSP\Task2 Equalizer\TASK 2')
        self.image = cv.cvtColor(cv.imread(self.filePath[0]),cv.COLOR_BGR2GRAY)
        self.ui.InputImage1.show()
        self.ui.InputImage1.setImage(self.image.T)

        self.dft = np.fft.fft2(self.image)
        self.real = np.real(self.dft)
        self.imaginary = np.imag(self.dft)
        self.magnitude = np.abs(self.dft)
        self.phase = np.angle(self.dft)

    def ChooseOperation(self):
        if (str(self.ui.Combo1.currentText())) == 'FT Magnitude':
            self.ui.FourierInput1.show()
            self.ui.FourierInput1.setImage(np.fft.fftshift(20 * np.log(self.magnitude.T)))
        elif (str(self.ui.Combo1.currentText())) == 'FT Phase':
            self.ui.FourierInput1.show()
            self.ui.FourierInput1.setImage(self.phase.T)
        elif (str(self.ui.Combo1.currentText())) == 'FT Real Component':
            self.ui.FourierInput1.show()
            self.ui.FourierInput1.setImage(20 * np.log(self.real.T))
        elif (str(self.ui.Combo1.currentText())) == 'FT Imaginary Component':
            self.ui.FourierInput1.show()
            self.ui.FourierInput1.setImage(self.imaginary.T)
        if (str(self.ui.Combo1.currentText())) == 'Select an Option':
            print('Please select a proper option')

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()
if __name__ == "__main__":
    main()