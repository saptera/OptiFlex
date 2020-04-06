import sys
import os
import numpy as np
from gui.setcrt_gui_desg import *

# Parallel processing settings
workers = 8
# Dataset list settings
trn_lst = str()
trn_flg = True
vld_lst = str()
vld_flg = True
tst_lst = str()
tst_flg = True
# Saving settings
out_dir = str()
size = (256, 256)
keep_dum = False
force_tst = True
# Tagging settings
lbl_std = []
group_list = []
# Label feature settings
js_type = True
hm_type = False
lbl_dummy = None
peak = 16.0
hm_seq = False
# Augmentation settings
aug_nr = 8
rnd_flp = True
axis = [-1, 0, 1]
rnd_rot = True
angle = (-10, 10)


class DataCreationWindow(QtWidgets.QMainWindow, Ui_DatCrt):
    def __init__(self, parent=None):
        super(DataCreationWindow, self).__init__(parent)
        self.setupUi(self)

        # Dataset creation toggle buttons
        self.TrnBtnGrp.buttonClicked.connect(self.set_trn_flag)
        self.VldBtnGrp.buttonClicked.connect(self.set_vld_flag)
        self.TstBtnGrp.buttonClicked.connect(self.set_tst_flag)

        # Dataset CSV file selection buttons
        self.TrnSetCSVBtn.clicked.connect(self.trn_lst_sel)
        self.VldSetCSVBtn.clicked.connect(self.vld_lst_sel)
        self.TstSetCSVBtn.clicked.connect(self.tst_lst_sel)

        # Output controllers
        self.OutDirBtn.clicked.connect(self.out_dir_sel)
        self.LblTypBtnGrp.buttonClicked.connect(self.set_lbl_typ)

        # Random flipping axes validation
        self.RndFlpX.clicked.connect(self.rnd_flp_x_vld)
        self.RndFlpY.clicked.connect(self.rnd_flp_y_vld)
        self.RndFlpXY.clicked.connect(self.rnd_flp_xy_vld)

        # Random rotation angle validation
        self.RndRotMin.valueChanged.connect(self.rnd_rot_agmin_vld)
        self.RndRotMax.valueChanged.connect(self.rnd_rot_agmax_vld)

        # Main control buttons
        self.StartButton.clicked.connect(self.start_ctrl)
        self.ExitButton.clicked.connect(self.exit_ctrl)

    # Dataset creation toggle buttons ------------------------------------------------------------------------------
    def set_trn_flag(self):
        global trn_flg, vld_flg, tst_flg, hm_type
        trn_flg = True if self.TrnBtnGrp.checkedId() == -2 else False    # TrnRdbY: Id = -2
        # Enable/Disable HM sequence setting
        hm_seq_enable = trn_flg and hm_type
        self.HMSeqLbl.setEnabled(hm_seq_enable)
        self.HMSeqCkb.setEnabled(hm_seq_enable)
        # Avoid void set creation
        if not (trn_flg or vld_flg or tst_flg):
            QtWidgets.QMessageBox.critical(self, "Error", "Must create at least one dataset type!")
            # Set button status back
            self.TrnRdbY.click()
            trn_flg = True

    def set_vld_flag(self):
        global trn_flg, vld_flg, tst_flg
        vld_flg = True if self.VldBtnGrp.checkedId() == -2 else False    # VldRdbY: Id = -2
        # Avoid void set creation
        if not (trn_flg or vld_flg or tst_flg):
            QtWidgets.QMessageBox.critical(self, "Error", "Must create at least one dataset type!")
            # Set button status back
            self.VldRdbY.click()
            vld_flg = True

    def set_tst_flag(self):
        global trn_flg, vld_flg, tst_flg
        tst_flg = True if self.TstBtnGrp.checkedId() == -2 else False    # TstRdbY: Id = -2
        # Avoid void set creation
        if not (trn_flg or vld_flg or tst_flg):
            QtWidgets.QMessageBox.critical(self, "Error", "Must create at least one dataset type!")
            # Set button status back
            self.TstRdbY.click()
            tst_flg = True

    # Dataset CSV file selection buttons ---------------------------------------------------------------------------
    def trn_lst_sel(self):
        lst_in, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Training Set List", filter="Set List (*.csv)")
        self.TrnSetCSV.setText(lst_in)

    def vld_lst_sel(self):
        lst_in, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Validation Set List", filter="Set List (*.csv)")
        self.VldSetCSV.setText(lst_in)

    def tst_lst_sel(self):
        lst_in, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Test Set List", filter="Set List (*.csv)")
        self.TstSetCSV.setText(lst_in)

    # Output controllers -------------------------------------------------------------------------------------------
    def out_dir_sel(self):
        out_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        self.OutDir.setText(out_path)

    def set_lbl_typ(self):
        global js_type, hm_type, trn_flg
        js_type = True if self.LblTypBtnGrp.checkedId() == -2 else False    # LblTypJS: Id = -2
        hm_type = True if self.LblTypBtnGrp.checkedId() == -3 else False    # LblTypHM: Id = -3
        # Enable/Disable HM sequence setting
        hm_seq_enable = hm_type and trn_flg
        self.HMSeqLbl.setEnabled(hm_seq_enable)
        self.HMSeqCkb.setEnabled(hm_seq_enable)

    # Random flipping axes validation ------------------------------------------------------------------------------
    def rnd_flp_x_vld(self):
        axis_x_bool = self.RndFlpX.isChecked()
        axis_y_bool = self.RndFlpY.isChecked()
        axis_xy_bool = self.RndFlpXY.isChecked()
        if not (axis_x_bool or axis_y_bool or axis_xy_bool):
            QtWidgets.QMessageBox.critical(self, "Error", "Must select at least one axis!")
            self.RndFlpX.click()

    def rnd_flp_y_vld(self):
        axis_x_bool = self.RndFlpX.isChecked()
        axis_y_bool = self.RndFlpY.isChecked()
        axis_xy_bool = self.RndFlpXY.isChecked()
        if not (axis_x_bool or axis_y_bool or axis_xy_bool):
            QtWidgets.QMessageBox.critical(self, "Error", "Must select at least one axis!")
            self.RndFlpY.click()

    def rnd_flp_xy_vld(self):
        axis_x_bool = self.RndFlpX.isChecked()
        axis_y_bool = self.RndFlpY.isChecked()
        axis_xy_bool = self.RndFlpXY.isChecked()
        if not (axis_x_bool or axis_y_bool or axis_xy_bool):
            QtWidgets.QMessageBox.critical(self, "Error", "Must select at least one axis!")
            self.RndFlpXY.click()

    # Random rotation angle validation -----------------------------------------------------------------------------
    def rnd_rot_agmin_vld(self):
        rnd_ag_min = self.RndRotMin.value()
        rnd_ag_max = self.RndRotMax.value()
        if rnd_ag_min > rnd_ag_max:
            self.RndRotMax.setValue(rnd_ag_min)

    def rnd_rot_agmax_vld(self):
        rnd_ag_max = self.RndRotMax.value()
        rnd_ag_min = self.RndRotMin.value()
        if rnd_ag_max < rnd_ag_min:
            self.RndRotMin.setValue(rnd_ag_max)

    # Main control buttons -----------------------------------------------------------------------------------------
    def start_ctrl(self):
        global trn_lst, trn_flg, vld_lst, vld_flg, tst_lst, tst_flg, lbl_std, group_list, keep_dum, force_tst, workers,\
            js_type, hm_type, hm_seq, lbl_dummy, aug_nr, rnd_flp, axis, rnd_rot, angle, peak, size, out_dir
        err_flg = False
        err_msg = str()

        # Get multiprocess cores
        workers = self.ProcNoSpb.value()

        # Set information --------------------------------------------------------------------------------------
        # Get training set information
        if trn_flg:
            # Verify file input
            trn_lst = self.TrnSetCSV.text()
            if not os.path.isfile(trn_lst):
                err_flg = True
                err_msg += "Invalid training set CSV list!\n"
            aug_nr = self.AugNoSpb.value()
            # Random flipping parameters
            rnd_flp = self.RndFlpCkb.isChecked()
            if rnd_flp:
                axis_xy = -1 if self.RndFlpXY.isChecked() else None
                axis_x = 0 if self.RndFlpX.isChecked() else None
                axis_y = 1 if self.RndFlpY.isChecked() else None
                axis = list(sorted(filter(None.__ne__, {axis_xy, axis_x, axis_y})))
            else:
                axis = []
            # Random rotation parameters
            rnd_rot = self.RndRotCkb.isChecked()
            if rnd_rot:
                angle = (self.RndRotMin.value(), self.RndRotMax.value())
            else:
                angle = (.0, .0)
        else:
            trn_lst = str()
            aug_nr = 0
            rnd_flp = rnd_rot = False
            axis = []
            angle = (.0, .0)

        # Get validation set information
        if vld_flg:
            # Verify file input
            vld_lst = self.VldSetCSV.text()
            if not os.path.isfile(vld_lst):
                err_flg = True
                err_msg += "Invalid validation set CSV list!\n"
        else:
            vld_lst = str()

        # Get validation set information
        if tst_flg:
            # Verify file input
            tst_lst = self.TstSetCSV.text()
            if not os.path.isfile(tst_lst):
                err_flg = True
                err_msg += "Invalid test set CSV list!\n"
        else:
            tst_lst = str()

        # IO information ---------------------------------------------------------------------------------------
        # Get saving options
        out_dir = self.OutDir.text()
        if not os.path.isdir(out_dir):
            err_flg = True
            err_msg += "Invalid output directory!\n"
        size = (self.ImgW.value(), self.ImgH.value())
        keep_dum = self.KpDumCkb.isChecked()
        force_tst = self.FrcTstCkb.isChecked()

        # Get label joint names definitions
        if len(self.LblLst.text()) == 0:
            err_flg = True
            err_msg += "Missing label joint list!\n"
            lbl_std = []
        else:
            lbl_std = [s.lstrip().rstrip() for s in self.LblLst.text().split(",")]
        # Get image/label group definitions
        if len(self.GrpLst.text()) == 0:
            group_list = None
        else:
            group_list = [s.lstrip().rstrip() for s in self.GrpLst.text().split(",")]

        # Get label feature options
        if js_type:
            lbl_dummy = {'left': None, 'top': None, 'width': 0, 'height': 0, 'label': None}
            peak = .0
        elif hm_type:
            lbl_dummy = np.zeros(size, np.float64)
            peak = self.HMPeak.value()
        # Get sequential option
        if trn_flg and hm_type:
            hm_seq = self.HMSeqCkb.isChecked()
        else:
            hm_seq = False

        # Finalize ---------------------------------------------------------------------------------------------
        if err_flg:
            QtWidgets.QMessageBox.critical(self, "Error", err_msg)
            return
        # Write to parameter file
        with open("gui/setcrt_gui_glbv.py", "w") as glb:
            glb.write("def init():\n")
            glb.write("    global trn_lst, trn_flg, vld_lst, vld_flg, tst_lst, tst_flg, lbl_std, group_list, keep_dum, "
                      "force_tst, workers, \\\n           js_type, hm_type, hm_seq, lbl_dummy, aug_nr, rnd_flp, axis, r"
                      "nd_rot, angle, peak, size, out_dir\n")
            glb.write("    # Parallel processing settings\n")
            glb.write("    workers = %d\n" % workers)
            glb.write("    # Dataset list settings\n")
            glb.write("    trn_lst = '%s'\n" % trn_lst)
            glb.write("    trn_flg = %s\n" % trn_flg)
            glb.write("    vld_lst = '%s'\n" % vld_lst)
            glb.write("    vld_flg = %s\n" % vld_flg)
            glb.write("    tst_lst = '%s'\n" % tst_lst)
            glb.write("    tst_flg = %s\n" % tst_flg)
            glb.write("    # Saving settings\n")
            glb.write("    out_dir = '%s'\n" % out_dir)
            glb.write("    size = (%d, %d)\n" % (size[0], size[1]))
            glb.write("    keep_dum = %s\n" % keep_dum)
            glb.write("    force_tst = %s\n" % force_tst)
            glb.write("    # Tagging settings\n")
            glb.write("    lbl_std = %s\n" % lbl_std)
            glb.write("    group_list = %s\n" % group_list)
            glb.write("    # Label feature settings\n")
            glb.write("    js_type = %s\n" % js_type)
            glb.write("    hm_type = %s\n" % hm_type)
            glb.write("    lbl_dummy = None\n")
            glb.write("    peak = %f\n" % peak)
            glb.write("    hm_seq = %s\n" % hm_seq)
            glb.write("    # Augmentation settings\n")
            glb.write("    aug_nr = %d\n" % aug_nr)
            glb.write("    rnd_flp = %s\n" % rnd_flp)
            glb.write("    axis = %s\n" % axis)
            glb.write("    rnd_rot = %s\n" % rnd_rot)
            glb.write("    angle = (%f, %f)\n" % (angle[0], angle[1]))
        self.destroy()
        QtCore.QCoreApplication.instance().quit()

    def exit_ctrl(self):
        self.destroy()
        print("Exit system.")
        sys.exit()


# Main Function ------------------------------------------------------------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = DataCreationWindow()
    window.show()
    app.exec_()
