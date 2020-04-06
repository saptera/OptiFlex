import sys
import os
import copy
import csv
import cv2
from OptiFlex.gui.manlbl_gui_desg import *
from OptiFlex.data.vid_proc import get_frm
from OptiFlex.data.lbl_proc import jsl_read, jsl_write


# Global definitions  -----------------------------------------------------------------------------------------------  #
# Define a class to handel cross window signals
class ComSig(QtCore.QObject):
    # Control signals
    frm_ctrl_sig = QtCore.pyqtSignal(int)  # Frame selection operation synchronizer
    lbl_ctrl_sig = QtCore.pyqtSignal(int)  # Label selection operation synchronizer
    # Data communication signals
    frm_info_sig = QtCore.pyqtSignal(int, str)  # Current frame information messenger
    lbl_tag_sig = QtCore.pyqtSignal(str)  # Label combo box / table setup messenger
    lbl_dat_sig = QtCore.pyqtSignal(int, str, int, int, int, int, tuple)  # Current label data messenger
    lbl_info_sig = QtCore.pyqtSignal(int, str)  # Current label information messenger


# Define global variables
com_sig = ComSig()  # Global signals
vid_cap = cv2.VideoCapture()  # OpenCV video capture
out_dir = str()  # Manual label output directory
color_lst = []  # Color list for labels
loaded = False  # File loading status flag
frm = None  # Current frame image
lbl_dat = {}  # Manual labelled label data
saving = False  # Saving status flag
frm_msg = str()  # Information message of current frame
lbl_msg = "Current label: [None]"  # Information message of current label


def manlbl_file_io(idx):
    # Set file names
    frm_file = os.path.join(out_dir, ("frm_%05d.png" % idx))
    lbl_file = os.path.join(out_dir, ("frm_%05d.json" % idx))
    # Write image, write when file is not exist
    if not os.path.isfile(frm_file):
        cv2.imwrite(os.path.join(out_dir, frm_file), frm)
    # Write label
    lst_lbl = []  # INIT VAR
    for i in range(len(lbl_dat)):
        lst_lbl.append(copy.deepcopy(lbl_dat[i]))
    jsl_write(os.path.join(out_dir, lbl_file), lst_lbl)


# [ControlViewer] Manual labelling main controller  -----------------------------------------------------------------  #
class ControlViewer(QtWidgets.QMainWindow, Ui_ControlViewer):
    tot_frm = 1  # Total frames to label
    cnt_frm = 0  # Frame currently labelling (Not the frame number)

    # [ControlViewer] main functions  ----------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(ControlViewer, self).__init__(parent)
        self.setupUi(self)
        self.frame_selector_limiter()
        self.statusBar.showMessage("Ready!")
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        # Set file saving flag contorol
        self.savemodeGroup.buttonClicked.connect(self.set_saving_flag)
        # Set frame control short cut keys
        self.prevButton.setShortcut("A")
        self.nextButton.setShortcut("D")
        # Update frame selection controls
        self.frameInitial.valueChanged['int'].connect(self.frame_selection_update)
        self.frameStep.valueChanged['int'].connect(self.frame_selection_update)
        self.frameFinal.valueChanged['int'].connect(self.frame_selection_update)
        self.frmselButton.clicked.connect(self.frame_selection_update)
        # Slider value control
        self.frameSlider.valueChanged['int'].connect(self.frame_slider_control)
        # Signal receiver
        com_sig.frm_info_sig.connect(self.status_report)
        com_sig.lbl_info_sig.connect(self.status_report)
        com_sig.frm_ctrl_sig.connect(self.frame_signal)
        com_sig.lbl_tag_sig.connect(self.setup_id_combo_box)
        com_sig.lbl_dat_sig.connect(self.label_dat_signal)
        # Exit button
        self.exitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)

    # Custom signals between different windows
    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            if event.key() == QtCore.Qt.Key_A:
                com_sig.frm_ctrl_sig.emit(-1)
            elif event.key() == QtCore.Qt.Key_D:
                com_sig.frm_ctrl_sig.emit(1)
            elif event.key() == QtCore.Qt.Key_W:
                com_sig.lbl_ctrl_sig.emit(-1)
            elif event.key() == QtCore.Qt.Key_S:
                com_sig.lbl_ctrl_sig.emit(1)
            else:
                pass

    def set_saving_flag(self):
        global saving, sv_frm_no
        saving = True if self.savemodeGroup.checkedId() == -3 else False  # Record Mode: Id = -3
        sv_frm_no = self.frameSlider.value()

    def status_report(self):
        self.statusBar.showMessage(lbl_msg + frm_msg)

    # [ControlViewer] Frame controller and signals  --------------------------------------------------------------------
    def frame_selector_limiter(self):
        self.frameFinal.setMinimum(self.frameInitial.value() + 1)
        self.frameStep.setMaximum(self.frameFinal.value() - self.frameInitial.value())

    def frame_selection_update(self):
        # Set slider
        self.frameSlider.setRange(self.frameInitial.value(), self.frameFinal.value())
        self.frameSlider.setSingleStep(self.frameStep.value())
        # Set slider value indicator
        self.sliderValue.setRange(self.frameInitial.value(), self.frameFinal.value())
        self.sliderValue.setSingleStep(self.frameStep.value())
        # Calculate total frames to be labelled
        self.tot_frm = (self.frameFinal.value() - self.frameInitial.value()) // self.frameStep.value() + 1
        # Check current value
        self.frame_selector_limiter()
        self.frame_slider_control()

    def frame_slider_control(self):
        global frm_msg
        # Verify and set slider value
        sld_diff = self.frameSlider.value() - self.frameInitial.value()
        sld_step = self.frameStep.value()
        self.cnt_frm = sld_diff // sld_step
        if sld_diff % sld_step != 0:
            self.frameSlider.setValue(self.frameInitial.value() + sld_step * self.cnt_frm)
        # Signalling frame information
        frm_msg = " | Current frame: %d | Progress: %d/%d - %.2f%%" \
            % (self.frameSlider.value(), self.cnt_frm + 1, self.tot_frm, (self.cnt_frm + 1) / self.tot_frm * 100)
        com_sig.frm_info_sig.emit(self.frameSlider.value(), frm_msg)

    def frame_signal(self, val):
        # File operation
        if saving:
            manlbl_file_io(self.frameSlider.value())
        # Control signal
        if val == -1:
            self.sliderValue.stepDown()
        elif val == 1:
            self.sliderValue.stepUp()
        else:
            pass

    # [ControlViewer] Label controller and signals  --------------------------------------------------------------------
    def setup_id_combo_box(self, val):
        self.idBox.addItem(val)

    def label_dat_signal(self, val_i, val_t, val_x, val_y, val_w, val_h, val_c):
        self.idBox.setCurrentIndex(val_i + 1)
        self.xValue.setValue(val_x)
        self.yValue.setValue(val_y)
        self.wValue.setValue(val_w)
        self.hValue.setValue(val_h)


# [FrameViewer] Manual labelling frame display and manual labeler ---------------------------------------------------  #
# [FrameViewer] Label item definition  ---------------------------------------------------------------------------------
class LabelItem(QtWidgets.QGraphicsRectItem):
    # Define local geometry reporting variables
    item_x = 0
    item_y = 0
    item_w = 0
    item_h = 0

    def __init__(self, x, y, w, h, ltg, lid, color, resize=False, parent=None):
        """ Custom label rectangle item.
        Args:
            x (int): Label X coordinate.
            y (int): Label Y coordinate.
            w (int): Label width.
            h (int): Label height.
            ltg (str): Label name.
            lid (int): Label sequential ID.
            color (tuple[int, int, int]): Label RGB color code.
            resize (bool): Flag for resize allowance (default: False).
            parent (None): None.
        """
        super(LabelItem, self).__init__(parent)
        # Set item properties
        self.setData(0, lid)
        self.setData(1, ltg)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        # Set frame of item
        pen = QtGui.QPen()
        pen.setWidth(1)
        pen.setColor(QtGui.QColor(color[0], color[1], color[2]))
        self.setPen(pen)
        # Set item geometry
        self.setPos(x, y)
        self.setRect(- w / 2, - h / 2, w, h)
        self.setZValue(16)  # Layer 16 for labels

    # Define a UserType for label items
    def type(self):
        return QtWidgets.QGraphicsRectItem.UserType + 1

    def itemChange(self, change, value):
        global color_lst, lbl_msg, lbl_dat
        # Generate a report when label item changed in position or size
        if (self.item_x != self.pos().x()) or (self.item_y != self.pos().y()) \
                or (self.item_w != self.rect().width()) or (self.item_h != self.rect().height()):
            self.item_x = self.pos().x()
            self.item_y = self.pos().y()
            self.item_w = self.rect().width()
            self.item_h = self.rect().height()
            if self.data(0) != -1:
                # Report string for status bar
                lbl_msg = "Current label: %s - Position (%d, %d); Size (%d, %d)"\
                    % (self.data(1), self.item_x, self.item_y, self.item_w, self.item_h)
                com_sig.lbl_info_sig.emit(self.data(0), lbl_msg)
                # Report data signal for control and list windows
                com_sig.lbl_dat_sig.emit(self.data(0), self.data(1), int(self.item_x), int(self.item_y),
                                         int(self.item_w), int(self.item_h), color_lst[self.data(0)])
                # Report dictionary to pass label data to file
                lbl_dat[self.data(0)] = {"left": int(self.item_x - self.item_w / 2),
                                         "top": int(self.item_y - self.item_h / 2),
                                         "width": int(self.item_w), "height": int(self.item_h),
                                         "label": self.data(1)}
        return QtWidgets.QGraphicsRectItem.itemChange(self, change, value)


# [FrameViewer] Label scene definition  --------------------------------------------------------------------------------
class LabelScene(QtWidgets.QGraphicsScene):
    lbl_id = -1
    lbl_x = 0
    lbl_y = 0
    lbl_w = 0
    lbl_h = 0
    lbl_tg = str()
    lbl_c = (0, 0, 0)

    def __init__(self, parent=None):
        super(LabelScene, self).__init__(parent)
        com_sig.lbl_dat_sig.connect(self.set_label)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # Check and re-generate label when SHIFT is pressed (only ONE label with same ID is allowed)
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                point = event.scenePos()
                for i in self.items():
                    if self.lbl_id == i.data(0):
                        self.removeItem(i)
                        break
                item = LabelItem(point.x(), point.y(), self.lbl_w, self.lbl_h, self.lbl_tg, self.lbl_id, self.lbl_c)
                self.addItem(item)
            else:
                QtWidgets.QGraphicsScene.mousePressEvent(self, event)
        # RIGHT CLICK to delete all selected items
        elif event.button() == QtCore.Qt.RightButton:
            for i in self.selectedItems():
                # Report data signal for control and list windows
                com_sig.lbl_dat_sig.emit(i.data(0), i.data(1), -1, -1, int(i.boundingRect().width() - 1),
                                         int(i.boundingRect().height() - 1), color_lst[i.data(0)])
                # Report dictionary to pass label data to file
                lbl_dat[i.data(0)] = {"left": None, "top": None, "width": None, "height": None, "label": i.data(1)}
                # Remove item
                self.removeItem(i)
        else:
            QtWidgets.QGraphicsScene.mousePressEvent(self, event)

    def set_label(self, val_i, val_t, val_x, val_y, val_w, val_h, val_c):
        self.lbl_id = val_i
        self.lbl_tg = val_t
        self.lbl_x = val_x
        self.lbl_y = val_y
        self.lbl_w = val_w
        self.lbl_h = val_h
        self.lbl_c = val_c


# [FrameViewer] Label window definition  -------------------------------------------------------------------------------
class FrameViewer(QtWidgets.QMainWindow, Ui_FrameViewer):
    first_call = True

    def __init__(self, parent=None):
        super(FrameViewer, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        com_sig.frm_info_sig.connect(self.frame_loader)
        self.scene = LabelScene()
        # Status signal receiver
        com_sig.frm_info_sig.connect(self.status_report)
        com_sig.lbl_info_sig.connect(self.status_report)

    # Custom signals between different windows
    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            if event.key() == QtCore.Qt.Key_A:
                com_sig.frm_ctrl_sig.emit(-1)
            elif event.key() == QtCore.Qt.Key_D:
                com_sig.frm_ctrl_sig.emit(1)
            elif event.key() == QtCore.Qt.Key_W:
                com_sig.lbl_ctrl_sig.emit(-1)
            elif event.key() == QtCore.Qt.Key_S:
                com_sig.lbl_ctrl_sig.emit(1)
            else:
                QtWidgets.QMainWindow.keyPressEvent(self, event)
        else:
            QtWidgets.QMainWindow.keyPressEvent(self, event)

    def frame_loader(self, val, _):
        global frm
        if loaded:
            # Convert OpenCV nparray image to QImage
            frm = get_frm(vid_cap, val)
            lbl_img = QtGui.QImage(frm, frm.shape[1], frm.shape[0], frm.shape[1] * 3, QtGui.QImage.Format_RGB888)
            if self.first_call:
                # Setup QGraphicsScene
                self.scene.setSceneRect(0, 0, lbl_img.width(), lbl_img.height())
                # Setup QGraphicsView
                scroll_bar_size = self.frameDisplay.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
                self.frameDisplay.setFixedSize(lbl_img.width() + scroll_bar_size, lbl_img.height() + scroll_bar_size)
                self.frameDisplay.setScene(self.scene)
                # Change flag
                self.first_call = False
            else:
                # Delete previous frame, avoid memory leak
                for i in self.scene.items():
                    if i.type() == 7:  # QGraphicsPixmapItem::Type = 7
                        self.scene.removeItem(i)
            # Add new frame, send to background
            self.scene.addPixmap(QtGui.QPixmap(lbl_img.rgbSwapped()))
            for i in self.scene.items():
                if i.type() == 7:  # QGraphicsPixmapItem::Type = 7
                    i.setZValue(-1)

    def status_report(self):
        self.statusBar.showMessage(lbl_msg + frm_msg)


# [ListViewer] Manual labelling label list display and selection  ---------------------------------------------------  #
class ListViewer(QtWidgets.QMainWindow, Ui_ListViewer):
    curr_row = None  # Currently activated row

    # [ListViewer] main functions  ----------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(ListViewer, self).__init__(parent)
        self.setupUi(self)
        self.listTable.setSortingEnabled(False)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        # Set label list control short cut keys
        self.lblpreButton.setShortcut("W")
        self.lblnxtButton.setShortcut("S")
        # Set button controls
        self.lblpreButton.clicked.connect(self.label_prev)
        self.lblnxtButton.clicked.connect(self.label_next)
        # Signal communication
        com_sig.lbl_ctrl_sig.connect(self.label_signal)
        com_sig.lbl_dat_sig.connect(self.set_label)

    # Custom signals between different windows
    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            if event.key() == QtCore.Qt.Key_A:
                com_sig.frm_ctrl_sig.emit(-1)
            elif event.key() == QtCore.Qt.Key_D:
                com_sig.frm_ctrl_sig.emit(1)
            elif event.key() == QtCore.Qt.Key_W:
                com_sig.lbl_ctrl_sig.emit(-1)
            elif event.key() == QtCore.Qt.Key_S:
                com_sig.lbl_ctrl_sig.emit(1)
            else:
                QtWidgets.QMainWindow.keyPressEvent(self, event)
        else:
            QtWidgets.QMainWindow.keyPressEvent(self, event)

    # [ListViewer] Label list CSV file loader  -------------------------------------------------------------------------
    def load_list(self, csv_file):
        global color_lst
        with open(csv_file) as ll:
            lbl_list = csv.reader(ll)
            i = 0
            for row in lbl_list:
                self.listTable.insertRow(i)
                j = 0
                for dat in row:
                    # Load label name normally
                    if j == 0:
                        item = QtWidgets.QTableWidgetItem(dat)
                        self.listTable.setItem(i, j, item)
                        # Label name signalling
                        com_sig.lbl_tag_sig.emit(dat)
                    # Left coordinates X and Y in blank (j + 2)
                    elif j == 1 or j == 2:
                        item = QtWidgets.QTableWidgetItem(dat)
                        self.listTable.setItem(i, j + 2, item)
                    # Last column is color definition
                    elif j == 3:
                        color_hex = dat.lstrip('#')
                        color_lst.append(tuple(int(color_hex[h:h + 2], 16) for h in (0, 2, 4)))
                    j += 1
                i += 1
        # Set window width with data loaded
        self.listTable.resizeColumnsToContents()
        lv_totw = self.listTable.verticalHeader().width() \
            + self.listTable.horizontalHeader().length() \
            + self.listTable.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent) \
            + self.listTable.frameWidth() * 2
        self.listTable.setFixedWidth(lv_totw)
        self.setFixedWidth(lv_totw)

    # [ListViewer] Label controller and signals  -----------------------------------------------------------------------
    def label_prev(self):
        global color_lst, lbl_msg
        if self.curr_row is None:
            self.curr_row = 0
        else:
            # Loop over at top label
            self.curr_row = self.listTable.rowCount() - 1 if self.curr_row == 0 else self.curr_row - 1
        self.listTable.selectRow(self.curr_row)
        # Selected label data signalling
        l_sig_t = self.listTable.item(self.curr_row, 0).text()
        l_sig_x = -1 if self.listTable.item(self.curr_row, 1) is None \
            else int(self.listTable.item(self.curr_row, 1).text())
        l_sig_y = -1 if self.listTable.item(self.curr_row, 2) is None \
            else int(self.listTable.item(self.curr_row, 2).text())
        l_sig_w = int(self.listTable.item(self.curr_row, 3).text())
        l_sig_h = int(self.listTable.item(self.curr_row, 4).text())
        com_sig.lbl_dat_sig.emit(self.curr_row, l_sig_t, l_sig_x, l_sig_y, l_sig_w, l_sig_h, color_lst[self.curr_row])
        # Selected label information signalling
        lbl_msg = "Current label: %s" % l_sig_t
        com_sig.lbl_info_sig.emit(self.curr_row, lbl_msg)

    def label_next(self):
        global color_lst, lbl_msg
        if self.curr_row is None:
            self.curr_row = 0
        else:
            # Loop over at bottom label
            self.curr_row = 0 if self.curr_row == self.listTable.rowCount() - 1 else self.curr_row + 1
        self.listTable.selectRow(self.curr_row)
        # Selected label data signalling
        l_sig_t = self.listTable.item(self.curr_row, 0).text()
        l_sig_x = -1 if self.listTable.item(self.curr_row, 1) is None \
            else int(self.listTable.item(self.curr_row, 1).text())
        l_sig_y = -1 if self.listTable.item(self.curr_row, 2) is None \
            else int(self.listTable.item(self.curr_row, 2).text())
        l_sig_w = int(self.listTable.item(self.curr_row, 3).text())
        l_sig_h = int(self.listTable.item(self.curr_row, 4).text())
        com_sig.lbl_dat_sig.emit(self.curr_row, l_sig_t, l_sig_x, l_sig_y, l_sig_w, l_sig_h, color_lst[self.curr_row])
        # Selected label information signalling
        lbl_msg = "Current label: %s" % l_sig_t
        com_sig.lbl_info_sig.emit(self.curr_row, lbl_msg)

    def label_signal(self, var):
        if var == -1:
            self.label_prev()
        elif var == 1:
            self.label_next()
        else:
            pass

    def emit_label_signal(self):
        global color_lst, lbl_msg
        self.curr_row = self.listTable.currentRow()
        # Selected label data signalling
        l_sig_t = self.listTable.item(self.curr_row, 0).text()
        l_sig_x = -1 if self.listTable.item(self.curr_row, 1) is None \
            else int(self.listTable.item(self.curr_row, 1).text())
        l_sig_y = -1 if self.listTable.item(self.curr_row, 2) is None \
            else int(self.listTable.item(self.curr_row, 2).text())
        l_sig_w = int(self.listTable.item(self.curr_row, 3).text())
        l_sig_h = int(self.listTable.item(self.curr_row, 4).text())
        com_sig.lbl_dat_sig.emit(self.curr_row, l_sig_t, l_sig_x, l_sig_y, l_sig_w, l_sig_h, color_lst[self.curr_row])
        # Selected label information signalling
        lbl_msg = "Current label: %s" % l_sig_t
        com_sig.lbl_info_sig.emit(self.curr_row, lbl_msg)

    def set_label(self, val_i, val_t, val_x, val_y, val_w, val_h, val_c):
        self.listTable.setItem(val_i, 1, QtWidgets.QTableWidgetItem(str(val_x)))
        self.listTable.setItem(val_i, 2, QtWidgets.QTableWidgetItem(str(val_y)))
        self.listTable.setItem(val_i, 3, QtWidgets.QTableWidgetItem(str(val_w)))
        self.listTable.setItem(val_i, 4, QtWidgets.QTableWidgetItem(str(val_h)))


# [MainLoader] Manual labelling main loader  ------------------------------------------------------------------------  #
class MainLoader(QtWidgets.QMainWindow, Ui_MainLoader):
    def __init__(self, parent=None):
        super(MainLoader, self).__init__(parent)
        self.setupUi(self)
        self.controlWindow = ControlViewer()
        self.frameWindow = FrameViewer()
        self.listWindow = ListViewer()

        self.videoButton.clicked.connect(self.video_selection)
        self.labelButton.clicked.connect(self.label_selection)
        self.outputButton.clicked.connect(self.output_selection)
        self.loadButton.clicked.connect(self.loading_func)
        self.exitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)

        self.controlWindow.idBox.currentIndexChanged['int'].connect(self.label_sync_box2lst)
        self.listWindow.listTable.currentItemChanged.connect(self.label_sync_lst2box)

    # [MainLoader] File loading button functions  ----------------------------------------------------------------------
    def video_selection(self):
        vid_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video File",
                                                            filter="Videos (*.avi *.mp4 *.mov *.mpeg)")
        self.videoPath.setText(vid_name)

    def label_selection(self):
        lst_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Label List File", filter="Label list (*.csv)")
        self.labelPath.setText(lst_name)

    def output_selection(self):
        out_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        self.outputPath.setText(out_path)

    # [MainLoader] Label selection controller synchronizer  ------------------------------------------------------------
    def label_sync_box2lst(self):
        global lbl_msg
        if self.controlWindow.idBox.currentIndex() == 0:
            self.listWindow.listTable.clearSelection()
            com_sig.lbl_dat_sig.emit(-1, str(), 0, 0, 0, 0, (0, 0, 0))
            lbl_msg = "Current label: [None]"
            com_sig.lbl_info_sig.emit(-1, lbl_msg)
        else:
            self.listWindow.listTable.selectRow(self.controlWindow.idBox.currentIndex() - 1)
            self.listWindow.emit_label_signal()

    def label_sync_lst2box(self):
        self.controlWindow.idBox.setCurrentIndex(self.listWindow.listTable.currentRow() + 1)

    # [MainLoader] Main loading function  ------------------------------------------------------------------------------
    def loading_func(self):
        global vid_cap, out_dir, loaded
        # Verify GUI inputs
        flag = False
        err_msg = str()
        vid_name = self.videoPath.text()
        if not os.path.isfile(vid_name):
            flag = True
            err_msg += "Video file does not exist!\n"
        lst_name = self.labelPath.text()
        if not os.path.isfile(lst_name):
            flag = True
            err_msg += "Label list file does not exist!\n"
        out_path = self.outputPath.text()
        if not os.path.isdir(out_path):
            flag = True
            err_msg += "Invalid output path!\n"
        if flag:
            QtWidgets.QMessageBox.critical(self, "Error", err_msg)
            return
        # File loading
        else:
            out_dir = out_path
            self.hide()
            # Load control window
            self.controlWindow.show()
            # Load frame viewer and video
            self.frameWindow.show()
            vid_cap = cv2.VideoCapture(vid_name)
            # Send video frame information to controls
            self.controlWindow.xValue.setMaximum(int(vid_cap.get(3)))  # propId 3, CV_CAP_PROP_FRAME_WIDTH
            self.controlWindow.wValue.setMaximum(int(vid_cap.get(3)))  # propId 3, CV_CAP_PROP_FRAME_WIDTH
            self.controlWindow.yValue.setMaximum(int(vid_cap.get(4)))  # propId 4, CV_CAP_PROP_FRAME_HEIGHT
            self.controlWindow.hValue.setMaximum(int(vid_cap.get(4)))  # propId 4, CV_CAP_PROP_FRAME_HEIGHT
            self.controlWindow.frameInitial.setMaximum(int(vid_cap.get(7) - 2))  # propId 7, CV_CAP_PROP_FRAME_COUNT
            self.controlWindow.frameFinal.setMaximum(int(vid_cap.get(7)) - 1)  # propId 7, CV_CAP_PROP_FRAME_COUNT
            self.controlWindow.frameFinal.setValue(int(vid_cap.get(7)) - 1)  # propId 7, CV_CAP_PROP_FRAME_COUNT
            # Load label viewer and label list
            self.listWindow.show()
            self.listWindow.load_list(lst_name)
            # Set status, send signal
            loaded = True
            com_sig.frm_info_sig.emit(0, " | Current frame: 0 | Progress: 0 of %d 0.00%%)" % vid_cap.get(7))


# Main Function ------------------------------------------------------------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainLoader()
    window.show()
    sys.exit(app.exec_())
