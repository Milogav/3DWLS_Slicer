import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import math
import datetime
from vtk.util import numpy_support
#
# PVC_vFinal
#
def getFileNameFromPath(path):##this function returns the fileName and the fileDir from an input path string (of the kind fileDir/fileName)
  pos=path.rfind('/')
  directory=path[0:pos]
  filename=path[pos+1::]
  pos=filename.rfind('.')
  filename=filename[0:pos]
  return filename,directory;

class PVC_vFinal(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "PVC_vFinal" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Partial Volume Correction"]
    self.parent.dependencies = []
    self.parent.contributors = ["Miguel Lopez Gavilan (UC3M), Pablo Garcia Polo (URJC)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    This extension performs partial volume correction of fMRI arterial spin labeling sequences
    applying the 3D weighted least squares algorithm, based on the previous existing algorithm developed by Asllani et. Al
    """
#
# PVC_vFinalWidget
#

class PVC_vFinalWidget(ScriptedLoadableModuleWidget):

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #initialize the variables for storing the image nodes and paths
    self.moduleDir=os.path.dirname(os.path.abspath(__file__))##retrieve slicer's module path
    self.cbfNode=False
    self.t1Node=False
    self.altNode=False
    self.labelNode=False
    self.lutNode=False
    self.cbfPath=False
    self.t1Path=False
    self.altPath=False
    self.labelPath=False
    self.lutPath=False
###############################################################################################################################
    ###########################################    WIDGETS & CONNECTIONS
    
    #SPM standalone directory selector
    self.selectSpmPath=ctk.ctkPathLineEdit()
    self.selectSpmPath.filters = ctk.ctkPathLineEdit.Dirs
    parametersFormLayout.addRow("Path to SPM directory: ", self.selectSpmPath)

    #matlab compiler directory selector -> path to .../V713 (or other version)
    self.selectMcrPath=ctk.ctkPathLineEdit()
    self.selectMcrPath.filters = ctk.ctkPathLineEdit.Dirs
    parametersFormLayout.addRow("Path to matlab compiler runtime directory: ", self.selectMcrPath)

    #input ASL volume file selector
    self.cbfPathSelector = ctk.ctkPathLineEdit()
    self.cbfPathSelector.filters = ctk.ctkPathLineEdit.Files
    parametersFormLayout.addRow("ASL image volume file: ", self.cbfPathSelector)

    #input T1 volume file selector
    self.t1PathSelector = ctk.ctkPathLineEdit()
    self.t1PathSelector.filters = ctk.ctkPathLineEdit.Files
    parametersFormLayout.addRow("Anatomical MRI image volume file: ", self.t1PathSelector)

    # check box for coregistering the anatomical MRI to a proton density image instead of registering to the ASL image volume
    #it is assumed that the PD image volume has the same dimensions and is registered to the ASL image)
    self.pdCoreg = qt.QCheckBox()
    self.pdCoreg.checked = 0
    self.pdCoreg.setToolTip("If checked, the anatomical MRI will be coregistered with the (PD) introduced in the ctkPathLineEdit below this checkbox")
    parametersFormLayout.addRow("Use an alternative image volume for anatomical MRI coreg + reslice:", self.pdCoreg)

    #alternative coreg volume file selector
    self.altPathSelector = ctk.ctkPathLineEdit()
    self.altPathSelector.filters = ctk.ctkPathLineEdit.Files
    self.altPathSelector.setDisabled(True)
    parametersFormLayout.addRow("                                Alternative image volume file: ", self.altPathSelector)

    # check box for enable loading a labelled volume (.nii) of the anatomical volume. If the box is checked and a proper label volume is loaded
    # the 3DWLS algorithm is applied independently for each region and then the individual results are added up together, giving as output 3 image volumes as in the regular case but providing information
    # of the mean perfusion value of GM, WM and CSF tissue of each regions and the mean value of the whole region
    self.enableLabel = qt.QCheckBox()
    self.enableLabel.checked = 0
    self.enableLabel.setToolTip("If checked, performs the volume correction independently for each labelled region and provides information about the perfusion values of each of these regions")
    parametersFormLayout.addRow("Use a label volume to perform PVC independently for each region:", self.enableLabel)

    #label volume file selector
    self.labelPathSelector = ctk.ctkPathLineEdit()
    self.labelPathSelector.filters = ctk.ctkPathLineEdit.Files
    self.labelPathSelector.setDisabled(True)
    parametersFormLayout.addRow("                                Label image volume file: ", self.labelPathSelector)

    #color LUT  textfile selector
    self.LUTPathSelector = ctk.ctkPathLineEdit()
    self.LUTPathSelector.filters = ctk.ctkPathLineEdit.Files
    self.LUTPathSelector.setDisabled(True)
    parametersFormLayout.addRow("                                Color LUT .txt file: ", self.LUTPathSelector)

    #output directory selector
    self.outputDirSelector = ctk.ctkPathLineEdit()
    self.outputDirSelector.filters = ctk.ctkPathLineEdit.Dirs
    self.outputDirSelector.setCurrentPath(self.cbfPathSelector.currentPath)
    parametersFormLayout.addRow("Output directory: ", self.outputDirSelector)

    #Kernel size input in PIXELS
    self.kernelSizePixels = ctk.ctkCoordinatesWidget()
    self.kernelSizePixels.minimum = 1
    self.kernelSizePixels.coordinates="3,3,3"#for setting the coordinates manually in the code it is required to introduce them as a string with this format "x,y,z"
    self.kernelSizePixels.maximum = 13
    self.kernelSizePixels.decimals = 0
    self.kernelSizePixels.singleStep= 2
    parametersFormLayout.addRow("Kernel size in pixels (x,y,z): ", self.kernelSizePixels)

    #Kernel size input in mm display
    self.dispKernelSizeMM = qt.QLineEdit()
    self.dispKernelSizeMM.readOnly=True
    parametersFormLayout.addRow("                 ...in mm (x,y,z): ", self.dispKernelSizeMM)

    #Weighting kernel type input
    self.kernelTypeWidget = qt.QComboBox()
    self.kernelTypeWidget.insertItem(0,"Exponential")
    self.kernelTypeWidget.insertItem(1,"Distance")
    self.kernelTypeWidget.insertItem(2,"Gaussian")
    parametersFormLayout.addRow("Choose weighting kernel type: ", self.kernelTypeWidget)

    # check box for normalizing the output with a template     #
    self.applyNorm = qt.QCheckBox()
    self.applyNorm.checked = 0
    self.applyNorm.setToolTip("If checked, the output images are normalized with the selected template. Calculations over the outputs(mean,min,max,error...) will be applied to this normalized images")
    parametersFormLayout.addRow("Normalize the output images:", self.applyNorm)

    # check box for applying a gaussian smoothing to the output or not     #
    self.applySmooth = qt.QCheckBox()
    self.applySmooth.checked = 0
    self.applySmooth.setToolTip("If checked, a gaussian smoothing filter is applied to the output volumes, with the selected radius and standard deviations")
    parametersFormLayout.addRow("Apply VTK gaussian smoothing to the output:", self.applySmooth)

    #widget for introducing the vtkImageGaussianSmooth radius factor
    self.gaussianRadius = ctk.ctkCoordinatesWidget()
    self.gaussianRadius.minimum = 0.5
    self.gaussianRadius.decimals = 1
    self.gaussianRadius.singleStep= 0.5
    self.gaussianRadius.setDisabled(True)
    parametersFormLayout.addRow("                               Radius factors: ", self.gaussianRadius)

    #widget for introducing the vtkImageGaussianSmooth standard deviation
    self.gaussianDeviation = ctk.ctkCoordinatesWidget()
    self.gaussianDeviation.minimum = 0.5
    self.gaussianDeviation.decimals = 2
    self.gaussianDeviation.singleStep= 0.05
    self.gaussianDeviation.setDisabled(True)
    parametersFormLayout.addRow("                               Standard deviations: ", self.gaussianDeviation)

    ##if history file exists fill the corresponding path selectors with the last locations
    if os.path.isfile(self.moduleDir+"/pathHistory.txt"):
      history=open(self.moduleDir+"/pathHistory.txt","r")########WINDOWS OS INCOMPATIBLILITY
      self.selectSpmPath.setCurrentPath(history.readline().strip())#load last spm location as default if history textfile exists (spm path must be in the 1st line of the txt file)#load last spm location if history textfile exists
      self.selectMcrPath.setCurrentPath(history.readline().strip())#load last mcr location as default if history textfile exists (mcr path must be in the 2nd line of the txt file)
      history.close()

    # PVC (3dWLS algorithm) Apply Button
    self.applyPVC = qt.QPushButton("Apply 3DWLS")
    self.applyPVC.toolTip = "Run partial volume correction on the ASL image."
    self.applyPVC.enabled = False
    parametersFormLayout.addRow(self.applyPVC)

    # connections
    self.applyPVC.connect('clicked(bool)', self.onApplyPVC)
    self.cbfPathSelector.connect("currentPathChanged(const QString&)", self.onSelectCBF)
    self.t1PathSelector.connect("currentPathChanged(const QString&)", self.onSelectT1)
    self.altPathSelector.connect("currentPathChanged(const QString&)", self.onSelectAlt)
    self.labelPathSelector.connect("currentPathChanged(const QString&)", self.onSelectLabel)
    self.pdCoreg.connect('clicked(bool)', self.onSelectAlt)
    self.enableLabel.connect('clicked(bool)', self.onSelectLabel)
    self.kernelSizePixels.connect("coordinatesChanged(double*)",self.onSelectKernelDims)
    self.applySmooth.connect('clicked(bool)', self.onApplyFilter)

    # Add vertical spacer
    self.layout.addStretch(1)

  def cleanup(self):
    pass
#########################################################################################################################################################
  ################################# WIDGET UTILITIES
  def onSelectCBF(self):
    self.cbfPath=self.cbfPathSelector.currentPath
    if self.cbfNode:
      slicer.mrmlScene.RemoveNode(self.cbfNode)##delete previous loaded cbf image if exists
    [temp,self.cbfNode]=slicer.util.loadVolume(self.cbfPath,returnNode=True)#load new cbf image
    [name,temp]=getFileNameFromPath(self.cbfPath)#get file name
    self.cbfNode.SetName(name)##name the node as the filename. This is done by default by slicer when loading the volume but if we load a volume more than one time,
    ## slicer stores the names (equal) and references the last loaded one as: "filename"+"_1" or another number to get a unique name, even when the previous node has been removed
    ##by the previous code line we make sure that the volume gets its proper name even if the user loads it twice by mistake. This trick is done with all the input volumes
    if not(self.pdCoreg.isChecked()):
      pixel_size=np.array(self.cbfNode.GetSpacing())##retrieve pixel dimensions of the asl image if the alternative image for coregistration is not enabled
      pixel_size=pixel_size.astype(float)
      self.onSelectKernelDims()
    if self.t1PathSelector.currentPath:##if t1 file is also avalable, enable the button for running the algorithm
      self.applyPVC.enabled=True
    else:
      self.applyPVC.enabled=False
    if not(self.outputDirSelector.currentPath):
        #if asl image volume is provided and output path is empty, set by default the asl path as the output path
        [temp,outDir]=getFileNameFromPath(self.cbfPathSelector.currentPath)
        self.outputDirSelector.setCurrentPath(outDir)

  def onSelectT1(self):
    self.t1Path=self.t1PathSelector.currentPath
    if self.t1Node:
      slicer.mrmlScene.RemoveNode(self.t1Node)##delete previous loaded cbf image if exists
    [temp,self.t1Node]=slicer.util.loadVolume(self.t1Path,returnNode=True)#load new cbf image
    [name,temp]=getFileNameFromPath(self.t1Path)
    self.t1Node.SetName(name)
    if self.cbfPathSelector.currentPath:##if asl file is also avalable, enable the button for running the algorithm
      self.applyPVC.enabled=True
    else:
      self.applyPVC.enabled=False

  def onSelectAlt(self):
    check=self.pdCoreg.isChecked()
    self.altPath=self.altPathSelector.currentPath
    if check:
      self.altPathSelector.setDisabled(False)
    else:
      self.altPathSelector.setDisabled(True)
      if self.altNode:
        slicer.mrmlScene.RemoveNode(self.altNode)
        self.onSelectKernelDims()
    if check and self.altPath:
      ##if an alternative path is introduced or available and the checkbox is checked the alternative volume is loaded to the scene
      ##and pixel size from the asl image is replaced by the one of the alt.image (although it must be the same for the later algorithm performance)
      if self.altNode:
        slicer.mrmlScene.RemoveNode(self.altNode)
      [temp,self.altNode]=slicer.util.loadVolume(self.altPath,returnNode=True)
      [name,temp]=getFileNameFromPath(self.altPath)
      self.altNode.SetName(name)
      pixel_size=np.array(self.altNode.GetSpacing())##retrieve pixel dimensions of the alternative image
      pixel_size=pixel_size.astype(float)
      self.onSelectKernelDims()

  def onSelectLabel(self):
    check=self.enableLabel.isChecked()
    self.labelPath=self.labelPathSelector.currentPath
    self.lutPath=self.LUTPathSelector.currentPath
    if check:
      self.labelPathSelector.setDisabled(False)
      self.LUTPathSelector.setDisabled(False)
      if not self.lutPath:
        self.LUTPathSelector.setCurrentPath(self.moduleDir+"/FreeSurferColorLUT.txt")#set as default LUT the freesurfer color LUT contained in the txt file inside the module folder
    else:
      self.labelPathSelector.setDisabled(True)
      self.LUTPathSelector.setDisabled(True)
      if self.labelNode:
        slicer.mrmlScene.RemoveNode(self.labelNode)
      if self.lutNode:
        slicer.mrmlScene.RemoveNode(self.lutNode)
    if check and self.labelPath:
      if self.labelNode:
        slicer.mrmlScene.RemoveNode(self.labelNode)
      if self.lutNode:
        slicer.mrmlScene.RemoveNode(self.lutNode)
      [temp,self.labelNode]=slicer.util.loadLabelVolume(self.labelPath,returnNode=True)
      [name,temp]=getFileNameFromPath(self.labelPath)
      self.labelNode.SetName(name)
      [temp,self.lutNode]=slicer.util.loadColorTable(self.lutPath,returnNode=True)
      [name,temp]=getFileNameFromPath(self.lutPath)
      self.lutNode.SetName(name)
      lutID=self.lutNode.GetID()
      self.labelNode.GetDisplayNode().SetAndObserveColorNodeID(lutID)

  def onApplyFilter(self):
    if self.applySmooth.isChecked():
      self.gaussianRadius.setDisabled(False)
      self.gaussianDeviation.setDisabled(False)
    else:
      self.gaussianRadius.setDisabled(True)
      self.gaussianDeviation.setDisabled(True)

  def onSelectKernelDims(self):
    pixel_size=np.array([0,0,0])
    kernel_size=self.kernelSizePixels.coordinates#get the introduced kernel size
    kernel_size=map(int,kernel_size.split(','))
    if kernel_size[0]<3:
      kernel_size[0]=3
    if kernel_size[1]<3:
      kernel_size[1]=3
    if self.pdCoreg.isChecked()and self.altNode:
      pixel_size=np.array(self.altNode.GetSpacing())
    elif self.cbfNode:
      pixel_size=np.array(self.cbfNode.GetSpacing())
    if pixel_size.any():
      self.kernelSizePixels.coordinates=str(kernel_size[0])+","+str(kernel_size[1])+","+str(kernel_size[2])
      dispKernelMM=np.array(kernel_size)*pixel_size
      self.dispKernelSizeMM.setText("%.2f x %.2f x %.2f" % (dispKernelMM[0],dispKernelMM[1],dispKernelMM[2]))
      
######################################################################################################################
      ##############   EXTENSION EXECUTION
  def onApplyPVC(self):
    logic=PVC_vFinalLogic(self)
    #retrive slicer module location to load the batch template file for SPM coregistration and segmentation
    #for building a specific batch file for the input image volumes
    #self.moduleDir=os.path.dirname(os.path.abspath(__file__))
    jobtime=datetime.datetime.now().isoformat()##retrieve date and time of the job being executed in ISO format
    pos=jobtime.rfind('.')
    jobtime=jobtime[0:pos]
    outDir=self.outputDirSelector.currentPath+"/OutputPVC_"+jobtime
    os.makedirs(outDir)#create an output folder in the output directory choosen in which the output images will be stored

    spmDir=self.selectSpmPath.currentPath
    mcrDir=self.selectMcrPath.currentPath
    history=open(self.moduleDir+"/pathHistory.txt","w")#save the last spm and mcr paths used to be set as default in the next loading of the slicer module// writing permission deletes any already existing file with the same name
    history.write(spmDir+"\n"+mcrDir+"\n")#spm path will be contained in the first line of the txt file, while mcr path will be contained in the next line
    history.close()
    ########WINDOWS OS INCOMPATIBLILITY(next line)

    table=open(outDir+"/JOB_INFO.txt","w")#create an textfile with the information of the current PVC job in table format to be imported in excel or calc
    table.write("INPUT_CBF_FILE "+self.cbfPath+"\n")
    table.write("INPUT_ANATOMICAL_FILE "+self.t1Path+"\n")
    if self.pdCoreg.isChecked():
      table.write("INPUT_AltCoregReference_FILE "+self.altPathSelector.currentPath+"\n")
    else:
      table.write("INPUT_AltCoregReference_FILE NO\n")

    if self.enableLabel.isChecked():
      table.write("INPUT_LABEL_FILE "+self.labelPathSelector.currentPath+"\n")
      table.write("INPUT_colorLUT_TEXTFILE "+self.LUTPathSelector.currentPath+"\n\n")
    else:
      table.write("INPUT_LABEL_FILE NO\n\n")

    templateBatch=open(self.moduleDir+"/templateBatch.m","r")###the PVC module must contain the templateBatch.m file
    ##this template batch file is the regular batch file produced by SPM to coregister+reslice an anatomical volume to a
    #functional volume and then segment the resulting coregistered anatomical volume; with the first two lines erased (lines for loading the volumes)
    #this following lines for loading our specific volumes are added together to the code contained in the template batch to another batch file which is the one that will be executed.
    currentBatch=open(outDir+"/spmBatch.m","w")#writing permission deletes any already existing file with the same name

    if self.pdCoreg.isChecked() and self.altPath:         ##if an alternative image is provided and the checkbox is checked, the alternative image is used as SPM coregistration and reslice reference
      currentBatch.write("matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {'"+self.altPath+"'};\n")########WINDOWS OS INCOMPATIBLILITY
      coregPath=self.altPath
    else:            ##else it uses the asl image as SPM coreg+reslice reference image
      coregPath=self.cbfPath
      currentBatch.write("matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {'"+self.cbfPath+"'};\n")########WINDOWS OS INCOMPATIBLILITY
    currentBatch.write("matlabbatch{1}.spm.spatial.coreg.estwrite.source = {'"+self.t1Path+"'};\n")########WINDOWS OS INCOMPATIBLILITY
    currentBatch.write(templateBatch.read())
    currentBatch.write("matlabbatch{2}.spm.spatial.preproc.tissue(1).tpm = {'"+spmDir+"/spm12_mcr/spm12/tpm/TPM.nii,1'};\n")########WINDOWS OS INCOMPATIBLILITY
    currentBatch.write("matlabbatch{2}.spm.spatial.preproc.tissue(2).tpm = {'"+spmDir+"/spm12_mcr/spm12/tpm/TPM.nii,2'};\n")########WINDOWS OS INCOMPATIBLILITY
    currentBatch.write("matlabbatch{2}.spm.spatial.preproc.tissue(3).tpm = {'"+spmDir+"/spm12_mcr/spm12/tpm/TPM.nii,3'};\n")########WINDOWS OS INCOMPATIBLILITY
    currentBatch.write("matlabbatch{2}.spm.spatial.preproc.tissue(4).tpm = {'"+spmDir+"/spm12_mcr/spm12/tpm/TPM.nii,4'};\n")########WINDOWS OS INCOMPATIBLILITY
    currentBatch.write("matlabbatch{2}.spm.spatial.preproc.tissue(5).tpm = {'"+spmDir+"/spm12_mcr/spm12/tpm/TPM.nii,5'};\n")########WINDOWS OS INCOMPATIBLILITY
    currentBatch.write("matlabbatch{2}.spm.spatial.preproc.tissue(6).tpm = {'"+spmDir+"/spm12_mcr/spm12/tpm/TPM.nii,6'};\n")########WINDOWS OS INCOMPATIBLILITY
    templateBatch.close()
    if self.enableLabel.isChecked() and self.labelPath:#if label was enable and a path was provided, compute SPM coregistration of the label volume to the ASL or the alternative image
      currentBatch.write("\nmatlabbatch{3}.spm.spatial.coreg.estwrite.ref = {'"+coregPath+",1'};\n")
      currentBatch.write("matlabbatch{3}.spm.spatial.coreg.estwrite.source = {'"+self.labelPath+",1'};\n")
      labelBatch=open(self.moduleDir+"/labelBatch.m","r")
      currentBatch.write(labelBatch.read())
      labelBatch.close()
    currentBatch.close()
    #execute the batch file via command in the terminal
    terminalCommand=spmDir+"/run_spm12.sh"+" "+mcrDir+" batch "+outDir+"/spmBatch.m"########WINDOWS OS INCOMPATIBLILITY
    os.system(terminalCommand)#this produces as output the coregistered and segmented anatomical volumes, which are stored in the path of the asl and t1 image volumes
    
    #get the required filenames and directories for importing the volumes and getting the corresponding nodes
    [t1FileName,t1Dir]=getFileNameFromPath(self.t1Path)
    [cbfFileName,cbfDir]=getFileNameFromPath(self.cbfPath)

    #move SPM registration and segmentation volumes to the output folder and erase inner and outer skull segmentation images and the .mat file resulting from spm segmentation (not used)
    os.rename(t1Dir+"/r"+t1FileName+".nii",outDir+"/r"+t1FileName+".nii")
    os.rename(t1Dir+"/c1r"+t1FileName+".nii",outDir+"/SegGM.nii")
    os.rename(t1Dir+"/c2r"+t1FileName+".nii",outDir+"/SegWM.nii")
    os.rename(t1Dir+"/c3r"+t1FileName+".nii",outDir+"/SegCSF.nii")
    if self.applyNorm.isChecked():
       os.rename(t1Dir+"/y_r"+t1FileName+".nii",outDir+"/deformField.nii")
    else:
       os.remove(t1Dir+"/y_r"+t1FileName+".nii")
    os.remove(t1Dir+"/c4r"+t1FileName+".nii")
    os.remove(t1Dir+"/c5r"+t1FileName+".nii")
    os.remove(t1Dir+"/r"+t1FileName+"_seg8.mat")

    [temp,gmNode]=slicer.util.loadVolume(outDir+"/SegGM.nii",returnNode=True)#import segmented volumes resulting from the spm coregistration and segmentation and get nodes of the imported segmented volumes
    [temp,wmNode]=slicer.util.loadVolume(outDir+"/SegWM.nii",returnNode=True)########WINDOWS OS INCOMPATIBLILITY
    [temp,csfNode]=slicer.util.loadVolume(outDir+"/SegCSF.nii",returnNode=True)
    [temp,regT1Node]=slicer.util.loadVolume(outDir+"/r"+t1FileName+".nii",returnNode=True)

    self.cbfNode=slicer.util.getNode(cbfFileName)
    CBF=slicer.util.array(cbfFileName)###get the numeric arrays from the volume nodes
    seg_gm=slicer.util.array("SegGM")
    seg_wm=slicer.util.array("SegWM")
    seg_csf=slicer.util.array("SegCSF")
    orig_seg_gm=seg_gm
    orig_seg_wm=seg_wm
    orig_seg_csf=seg_csf
    voxel_size=np.array(self.cbfNode.GetSpacing())##retrieve voxel dimensions of the asl image
    voxel_size=voxel_size.astype(float)
    voxel_vol=np.prod(voxel_size)#calculate real voxel volume in mm3
    brain_mask=seg_gm+seg_wm+seg_csf
    orig_brain_mask=brain_mask##save the original segmentation images for computing the real volume of the regions in case of normalization and label image present
    gm_volume=seg_gm.sum()*voxel_vol
    wm_volume=seg_wm.sum()*voxel_vol
    csf_volume=seg_csf.sum()*voxel_vol
    total_volume=brain_mask.sum()*voxel_vol

    label_img=0
    if self.enableLabel.isChecked() and self.labelPath:
      [labelFileName,labelDir]=getFileNameFromPath(self.labelPath)
      #move coregistered label volume to the output folder
      os.rename(labelDir+"/r"+labelFileName+".nii",outDir+"/rLabelVol.nii")
      [lutFileName,temp]=getFileNameFromPath(self.labelPath)
      [temp,self.labelNode]=slicer.util.loadLabelVolume(outDir+"/rLabelVol.nii",returnNode=True)
      [temp,self.lutNode]=slicer.util.loadColorTable(self.lutPath,returnNode=True)
      self.labelNode.GetDisplayNode().SetAndObserveColorNodeID(self.lutNode.GetID())
      label_img=slicer.util.array("rLabelVol")

    kernel_size=self.kernelSizePixels.coordinates#get the introduced kernel size and weight type matrix
    kernel_size=kernel_size.split(',')
    kernel_mm=self.dispKernelSizeMM.text.replace(" x "," ")
    table.write("KERNEL_SIZE X Y Z\n")
    table.write("in_pixels "+kernel_size[0]+" "+kernel_size[1]+" "+kernel_size[2]+"\n")
    table.write("in_mm "+kernel_mm+"\n")

    kernel_size=map(int,kernel_size)
    weight_type=self.kernelTypeWidget.currentText
    table.write("WEIGHTING_MATRIX_TYPE "+weight_type+"\n\n")

    if self.pdCoreg.isChecked()and self.altNode:
      pixel_size=np.array(self.altNode.GetSpacing())
    else:
      pixel_size=np.array(self.cbfNode.GetSpacing())

    if not (self.enableLabel.isChecked() and self.labelPath):
      perfusion=logic.WLS3D(seg_gm,seg_wm,seg_csf,CBF,pixel_size,kernel_size,weight_type)#run the 3DWLS algorithm
    else:#run the 3DWLS algorithm for each label region independently and  store mean CBF values for each region
      LUT=open(self.lutPath,"r")#self.LUTPathSelector.currentPath
      labelID=list()
      labelName=list()
      for line in LUT:
        entry=line.split(None,2)
        if len(entry)>1:
          labelID.append(entry[0])#retieve all label numbers of the LUT and store them in a list
          labelName.append(entry[1])#retrive all label names of the LUT and store them in a list in the same positions as their label number
      LUT.close()
      labelValues=np.unique(label_img)#list of all label numbers present in our label image
      labelRegions=list()#list to populate with the corresponding label names of the label numbers present in out file
      for x in labelValues:
        try:
          pos=labelID.index(str(x))
          labelRegions.append(labelName[pos])
        except:
          labelRegions.append("Unknown_label")
      l=len(labelRegions)
      c=0
      out_gm=np.zeros(np.shape(CBF))
      out_wm=np.zeros(np.shape(CBF))
      out_csf=np.zeros(np.shape(CBF))
      out_total=np.zeros(np.shape(CBF))
      for label in labelValues:
        #print label
        label_mask=label_img==label
        #print("PROCESSING REGION "+str(c+1)+"/"+str(l)+":  "+labelRegions[c]+"\n")
        c=c+1
        perfusion=logic.WLS3D(seg_gm*label_mask,seg_wm*label_mask,seg_csf*label_mask,CBF*label_mask,pixel_size,kernel_size,weight_type)#run the 3DWLS algorithm for each label region
        out_gm=out_gm+perfusion.gm
        out_wm=out_wm+perfusion.wm
        out_csf=out_csf+perfusion.csf
        out_total=out_total+perfusion.total
      perfusion.gm=out_gm
      perfusion.wm=out_wm
      perfusion.csf=out_csf
      perfusion.total=out_total


    volumesLogic = slicer.modules.volumes.logic()
    #As the output volumes have the same pixel spacing, size... and similar intensity values as the asl image, we can create the new output volume nodes by cloning the existing cbf node
    volumesLogic.CloneVolume(slicer.mrmlScene, self.cbfNode, 'Perfusion_GM')
    volumesLogic.CloneVolume(slicer.mrmlScene, self.cbfNode, 'Perfusion_WM')
    volumesLogic.CloneVolume(slicer.mrmlScene, self.cbfNode, 'Perfusion_CSF')
    volumesLogic.CloneVolume(slicer.mrmlScene, self.cbfNode, 'Perfusion_TOTAL')#####ONLY FOR TEST (result of adding all individual perfusion maps obtained)
    ##casting of the clone volume to float in case ASL volume is of integer type (as we may have cloned an int16 volume array, when assigning the new float 32 volume we loose the decimals as the variable is of integer type)
    ##using an vtkImageCast object
    nodeOutGM=slicer.util.getNode('Perfusion_GM')
    nodeOutWM=slicer.util.getNode('Perfusion_WM')
    nodeOutCSF=slicer.util.getNode('Perfusion_CSF')
    nodeOutTOTAL=slicer.util.getNode('Perfusion_TOTAL')

    cast=vtk.vtkImageCast()
    cast.SetInputData(nodeOutGM.GetImageData())
    cast.SetOutputScalarTypeToDouble()
    cast.Update()
    nodeOutGM.SetAndObserveImageData(cast.GetOutput())

    cast=vtk.vtkImageCast()
    cast.SetInputData(nodeOutWM.GetImageData())
    cast.SetOutputScalarTypeToDouble()
    cast.Update()
    nodeOutWM.SetAndObserveImageData(cast.GetOutput())

    cast=vtk.vtkImageCast()
    cast.SetInputData(nodeOutCSF.GetImageData())
    cast.SetOutputScalarTypeToDouble()
    cast.Update()
    nodeOutCSF.SetAndObserveImageData(cast.GetOutput())

    cast=vtk.vtkImageCast()
    cast.SetInputData(nodeOutTOTAL.GetImageData())
    cast.SetOutputScalarTypeToDouble()
    cast.Update()
    nodeOutTOTAL.SetAndObserveImageData(cast.GetOutput())

    ## subsitute the volume arrays of the cloned output nodes by the volumes obtained after doing the 3DWLS

    dataOutGM=slicer.util.array('Perfusion_GM')
    dataOutGM[:]=perfusion.gm
    nodeOutGM.GetImageData().Modified()

    dataOutWM=slicer.util.array('Perfusion_WM')
    dataOutWM[:]=perfusion.wm
    nodeOutWM.GetImageData().Modified()

    dataOutCSF=slicer.util.array('Perfusion_CSF')
    dataOutCSF[:]=perfusion.csf
    nodeOutCSF.GetImageData().Modified()

    dataOutTOTAL=slicer.util.array('Perfusion_TOTAL')
    dataOutTOTAL[:]=perfusion.total
    nodeOutTOTAL.GetImageData().Modified()

    #save the output images as .nii volumes in the output folder specified before.These volumes correspond to the raw output from the algorithm i.e. without any normalization and/or smoothing
    #the new filenames will be the filename of the asl image file + _(tissue)Perfusion.nii || Tissues->gm=grey matter // wm=white matter // csf=cerebrospinal fluid
    slicer.util.saveNode(nodeOutGM,outDir+"/PerfGM_"+cbfFileName+".nii")########WINDOWS OS INCOMPATIBLILITY
    slicer.util.saveNode(nodeOutWM,outDir+"/PerfWM_"+cbfFileName+".nii")########WINDOWS OS INCOMPATIBLILITY
    slicer.util.saveNode(nodeOutCSF,outDir+"/PerfCSF_"+cbfFileName+".nii")########WINDOWS OS INCOMPATIBLILITY
    slicer.util.saveNode(nodeOutTOTAL,outDir+"/PerfTOTAL_"+cbfFileName+".nii")##ONLY FOR TEST (result of adding all individual perfusion maps obtained)###WINDOWS OS INCOMPATIBLILITY

    if self.applyNorm.isChecked():
    ##apply deformation field obtained in SPM segmentation if normalization checkbox is true
      table.write("OUTPUT_NORMALIZATION YES\n")

      ####modify the origin and directions on the images to normalize to match that of the registered t1 image (and therefore the same as the segmentations;
      #deformation field was obtained from the segmentation so the origin and direction should be changed to match their origin and direction
      origin=regT1Node.GetOrigin()
      dirMat=np.zeros([3,3])
      regT1Node.GetIJKToRASDirections(dirMat)
      self.cbfNode.SetOrigin(origin)
      nodeOutGM.SetIJKToRASDirections(dirMat)
      nodeOutGM.SetOrigin(origin)
      nodeOutGM.SetIJKToRASDirections(dirMat)
      nodeOutWM.SetOrigin(origin)
      nodeOutWM.SetIJKToRASDirections(dirMat)
      nodeOutCSF.SetOrigin(origin)
      nodeOutCSF.SetIJKToRASDirections(dirMat)
      nodeOutTOTAL.SetOrigin(origin)
      nodeOutTOTAL.SetIJKToRASDirections(dirMat)

      #save the origin and direction changes in the images
      slicer.util.saveNode(nodeOutGM,outDir+"/PerfGM_"+cbfFileName+".nii")########WINDOWS OS INCOMPATIBLILITY
      slicer.util.saveNode(nodeOutWM,outDir+"/PerfWM_"+cbfFileName+".nii")########WINDOWS OS INCOMPATIBLILITY
      slicer.util.saveNode(nodeOutCSF,outDir+"/PerfCSF_"+cbfFileName+".nii")########WINDOWS OS INCOMPATIBLILITY
      slicer.util.saveNode(nodeOutTOTAL,outDir+"/PerfTOTAL_"+cbfFileName+".nii")##ONLY FOR TEST (result of adding all individual perfusion maps obtained)###WINDOWS OS INCOMPATIBLILITY
      slicer.util.saveNode(self.cbfNode,outDir+"/mod"+cbfFileName+".nii")##ONLY FOR TEST (result of adding all individual perfusion maps obtained)###WINDOWS OS INCOMPATIBLILITY

      self.cbfPath="'"+outDir+"/mod"+cbfFileName+".nii';"
      self.t1Path="'"+outDir+"/r"+t1FileName+".nii';"
      perfGMpath="'"+outDir+"/PerfGM_"+cbfFileName+".nii';"
      perfWMpath="'"+outDir+"/PerfWM_"+cbfFileName+".nii';"
      perfCSFpath="'"+outDir+"/PerfCSF_"+cbfFileName+".nii';"
      perfTOTALpath="'"+outDir+"/PerfTOTAL_"+cbfFileName+".nii'"

      normBatch=open(outDir+"/normBatch.m","w")
      normBatch.write("matlabbatch{1}.spm.spatial.normalise.write.subj.def = {'"+outDir+"/deformField.nii'};\n")
      normBatch.write("matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {"+self.cbfPath+self.t1Path+perfGMpath+perfWMpath+perfCSFpath+perfTOTALpath+"};\n")
      normBatch.write("matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [NaN,NaN,NaN;NaN,NaN,NaN];\n")
      normBatch.write("matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2,2,4];\n")
      normBatch.write("matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;\n")
      normBatch.write("matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'norm';")
      normBatch.close()
      terminalCommand=spmDir+"/run_spm12.sh"+" "+mcrDir+" batch "+outDir+"/normBatch.m"########WINDOWS OS INCOMPATIBLILITY
      os.system(terminalCommand)

      os.remove(outDir+"/mod"+cbfFileName+".nii")
      os.rename(outDir+"/normmod"+cbfFileName+".nii",outDir+"/normOrigCBF.nii")
      #load the normalized volumes and override the variables storing the arrays and nodes by the new normalized volumes in order to carry out smoothing (if enabled) and the error, mean,max...calculations over the normalized volumes
      slicer.util.loadVolume(outDir+"/normOrigCBF.nii")
      
      os.rename(outDir+"/normPerfGM_"+cbfFileName+".nii",outDir+"/PerfGM_"+cbfFileName+".nii")
      os.rename(outDir+"/normPerfWM_"+cbfFileName+".nii",outDir+"/PerfWM_"+cbfFileName+".nii")
      os.rename(outDir+"/normPerfCSF_"+cbfFileName+".nii",outDir+"/PerfCSF_"+cbfFileName+".nii")
      os.rename(outDir+"/normPerfTOTAL_"+cbfFileName+".nii",outDir+"/PerfTOTAL_"+cbfFileName+".nii")
      [v,nodeOutGM]=slicer.util.loadVolume(outDir+"/PerfGM_"+cbfFileName+".nii",returnNode=True)
      [v,nodeOutWM]=slicer.util.loadVolume(outDir+"/PerfWM_"+cbfFileName+".nii",returnNode=True)
      [v,nodeOutCSF]=slicer.util.loadVolume(outDir+"/PerfCSF_"+cbfFileName+".nii",returnNode=True)
      [v,nodeOutTOTAL]=slicer.util.loadVolume(outDir+"/PerfTOTAL_"+cbfFileName+".nii",returnNode=True)
      CBF=slicer.util.array("normOrigCBF")

      perfusion.gm=slicer.util.array("PerfGM_"+cbfFileName)
      perfusion.wm=slicer.util.array("PerfWM_"+cbfFileName)
      perfusion.csf=slicer.util.array("PerfCSF_"+cbfFileName)
      perfusion.total=slicer.util.array("PerfTOTAL_"+cbfFileName)

      ##use the available default label image of the normalization template (contained in the same SPM folder as the normalization template)
      ##coregister the default label volume and the normalized segmentation volumes to have the same size as the normalized perfusion image.
      ##default labels and norm segmentation volumes must be in aligment with the normalized perfusion image prior to the coregistration 
      ##if a label image was provided before, it is overriden by the default normalized label image
      normLabelBatch=open(outDir+"/normLabelBatch.m","w")
      
      normLabelBatch.write("matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {'"+outDir+"/PerfTOTAL_"+cbfFileName+".nii"+"'};\n")##coregister label volume using nneighbor
      normLabelBatch.write("matlabbatch{1}.spm.spatial.coreg.estwrite.source = {'"+spmDir+"/spm12_mcr/spm12/tpm/labels_Neuromorphometrics.nii"+"'};\n")
      normLabelBatch.write("matlabbatch{2}.spm.spatial.coreg.estwrite.ref = {'"+outDir+"/PerfTOTAL_"+cbfFileName+".nii"+"'};\n")
      normLabelBatch.write("matlabbatch{2}.spm.spatial.coreg.estwrite.source = {'"+self.moduleDir+"/TPM_GM.nii'};\n")##coregister segmentation volumes using 4deg b-spline interpolation
      normLabelBatch.write("matlabbatch{2}.spm.spatial.coreg.estwrite.other = {'"+self.moduleDir+"/TPM_WM.nii';'"+self.moduleDir+"/TPM_CSF.nii'};\n")
      tempNormLabelBatch=open(self.moduleDir+"/normLabelBatch.m","r")
      normLabelBatch.write(tempNormLabelBatch.read())
      tempNormLabelBatch.close()
      normLabelBatch.close()
      terminalCommand=spmDir+"/run_spm12.sh"+" "+mcrDir+" batch "+outDir+"/normLabelBatch.m"########WINDOWS OS INCOMPATIBLILITY
      os.system(terminalCommand)

      table.write("CUSTOM_LABEL_VOLUME_INTRODUCED NO\n")
      table.write("USING_DEFAULT_NORMALIZATION_TEMPLATE_LABELS YES\n\n")
      
      [temp,normLabelNode]=slicer.util.loadLabelVolume(spmDir+"/spm12_mcr/spm12/tpm/rlabels_Neuromorphometrics.nii",returnNode=True)
      os.rename(spmDir+"/spm12_mcr/spm12/tpm/rlabels_Neuromorphometrics.nii",outDir+"/rNormLabels.nii")
      os.rename(self.moduleDir+"/rTPM_GM.nii",outDir+"/normSegGM.nii")
      os.rename(self.moduleDir+"/rTPM_WM.nii",outDir+"/normSegWM.nii")
      os.rename(self.moduleDir+"/rTPM_CSF.nii",outDir+"/normSegCSF.nii")
      slicer.util.loadVolume(outDir+"/normSegGM.nii")
      slicer.util.loadVolume(outDir+"/normSegWM.nii")
      slicer.util.loadVolume(outDir+"/normSegCSF.nii")
      seg_gm=slicer.util.array("normSegGM")
      seg_wm=slicer.util.array("normSegWM")
      seg_csf=slicer.util.array("normSegCSF")
      
      label_img=slicer.util.array("rlabels_Neuromorphometrics")
      self.lutPath=self.moduleDir+"/labelsNormTemplate.txt"
      [temp,self.lutNode]=slicer.util.loadColorTable(self.lutPath,returnNode=True)
      lutID=self.lutNode.GetID()
      normLabelNode.GetDisplayNode().SetAndObserveColorNodeID(lutID)

      LUT=open(self.lutPath,"r")
      labelID=list()
      labelName=list()
      for line in LUT:
          entry=line.split(None,2)
          if len(entry)>1:
            labelID.append(entry[0])#retieve all label numbers of the LUT and store them in a list
            labelName.append(entry[1])#retrive all label names of the LUT and store them in a list in the same positions as their label number
      LUT.close()
      labelValues=np.unique(label_img)#list of all label numbers present in our label image
      labelRegions=list()#list to populate with the corresponding label names of the label numbers present in out file
      for x in labelValues:
          try:
            pos=labelID.index(str(x))
            labelRegions.append(labelName[pos])
          except:
            labelRegions.append("Unknown_label")

    ##if smoothing checkbox is checked, apply the filter
    if not self.applySmooth.isChecked():
      table.write("GAUSSIAN_SMOOTHING NO\n\n")
    else:
      table.write("GAUSSIAN_SMOOTHING YES\n")
      radius=self.gaussianRadius.coordinates
      radius=map(float,radius.split(','))
      deviation=self.gaussianDeviation.coordinates
      deviation=map(float,deviation.split(','))
      table.write("Radius "+str(radius[0])+" "+str(radius[1])+" "+str(radius[2])+"\n")
      table.write("Standard_deviations "+str(deviation[0])+" "+str(deviation[1])+" "+str(deviation[2])+"\n\n")

      matrix = vtk.vtkMatrix4x4()
      nodeOutGM.GetIJKToRASMatrix(matrix)
      filter=vtk.vtkImageGaussianSmooth()
      filter.SetStandardDeviations(deviation)
      filter.SetRadiusFactors(radius)
      filter.SetDimensionality(3)
      filter.SetInputData(nodeOutGM.GetImageData())
      filter.Update()
      nodeOutGM.SetAndObserveImageData(filter.GetOutput())
      nodeOutGM.SetIJKToRASMatrix(matrix)

      matrix = vtk.vtkMatrix4x4()
      nodeOutWM.GetIJKToRASMatrix(matrix)
      filter=vtk.vtkImageGaussianSmooth()
      filter.SetStandardDeviations(deviation)
      filter.SetRadiusFactors(radius)
      filter.SetDimensionality(3)
      filter.SetInputData(nodeOutWM.GetImageData())
      filter.Update()
      nodeOutWM.SetAndObserveImageData(filter.GetOutput())
      nodeOutWM.SetIJKToRASMatrix(matrix)

      matrix = vtk.vtkMatrix4x4()
      nodeOutCSF.GetIJKToRASMatrix(matrix)
      filter=vtk.vtkImageGaussianSmooth()
      filter.SetStandardDeviations(deviation)
      filter.SetRadiusFactors(radius)
      filter.SetDimensionality(3)
      filter.SetInputData(nodeOutCSF.GetImageData())
      filter.Update()
      nodeOutCSF.SetAndObserveImageData(filter.GetOutput())
      nodeOutCSF.SetIJKToRASMatrix(matrix)

      matrix = vtk.vtkMatrix4x4()
      nodeOutTOTAL.GetIJKToRASMatrix(matrix)
      filter=vtk.vtkImageGaussianSmooth()
      filter.SetStandardDeviations(deviation)
      filter.SetRadiusFactors(radius)
      filter.SetDimensionality(3)
      filter.SetInputData(nodeOutTOTAL.GetImageData())
      filter.Update()
      nodeOutTOTAL.SetAndObserveImageData(filter.GetOutput())
      nodeOutTOTAL.SetIJKToRASMatrix(matrix)

      #save the output images as .nii volumes in the output folder .
      #These volumes correspond to the smoothed output from the algorithm. Overrides the raw image volumes or the normalized volumes (if normalization was enabled) with the smoothed output
      slicer.util.saveNode(nodeOutGM,outDir+"/PerfGM_"+cbfFileName+".nii")########WINDOWS OS INCOMPATIBLILITY
      slicer.util.saveNode(nodeOutWM,outDir+"/PerfWM_"+cbfFileName+".nii")########WINDOWS OS INCOMPATIBLILITY
      slicer.util.saveNode(nodeOutCSF,outDir+"/PerfCSF_"+cbfFileName+".nii")########WINDOWS OS INCOMPATIBLILITY
      slicer.util.saveNode(nodeOutTOTAL,outDir+"/PerfTOTAL_"+cbfFileName+".nii")##ONLY FOR TEST (result of adding all individual perfusion maps obtained)###WINDOWS OS INCOMPATIBLILITY
      perfusion.gm=slicer.util.array("PerfGM_"+cbfFileName)
      perfusion.wm=slicer.util.array("PerfWM_"+cbfFileName)
      perfusion.csf=slicer.util.array("PerfCSF_"+cbfFileName)
      perfusion.total=slicer.util.array("PerfTOTAL_"+cbfFileName)

    #compute root mean squared error between ASL input volume and the addition of the PVC output volumes (GM+WM+CSF i.e. TOTAL) to get a measure of the error (FOR TESTING)
    #save the error into the info textfile. Save the mean values of the original CBF and the perfusion volumes (mean values of the pixels that belong to their specific tissues->background and other tissue pixels are not taken into account)
    #save the approximate volumes of each tissue in mm3

    CBF.astype(float)
    brain_mask=seg_gm+seg_wm+seg_csf
    index=brain_mask>0#retrieve the indexes of the voxels that belong to brain tissue
    num_elem=brain_mask.sum()
    brain_mask[index]=1
    CBF=CBF*brain_mask#keep only the voxels belonging to brain tissue in the original CBF

    table.write("IMAGE_VOL MEAN_PERFUSION APROX_VOLUME\n")

    mean=str(np.sum(CBF)/num_elem)
    table.write("Original_CBF "+mean+" "+str(total_volume)+"\n")

    mean=str(np.sum(perfusion.total[index])/num_elem)
    table.write("Total_perfusion "+mean+" "+str(total_volume))
    error=CBF-perfusion.total
    error=np.absolute(error)
    error=np.sum(error)
    error=error/num_elem##divide accumulated error by the number of voxels corresponding to brain tissue (summation of the values of brain_mask) for doing the mean value without taking into account backgorund or other tissue (bone) voxels
    table.write(" (RMSE_error) "+str(error)+"\n")

    index=seg_gm>0#retrieve the indexes of the voxels belonging to grey matter
    num_elem=seg_gm[index].sum()##compute a weighted mean as a function of tissue probability: num_elem=index.sum()
    mean=str(np.sum(perfusion.gm[index])/num_elem)
    print("GM voxels: "+str(num_elem))
    table.write("GM_perfusion "+mean+" "+str(gm_volume)+"\n")##tissue volumes were calculated before in case normalization was performed, as this seg_gm would correspond to the normalized seg_gm

    index=seg_wm>0#retrieve the voxels indexes belonging to white matter
    num_elem=seg_wm[index].sum()
    mean=str(np.sum(perfusion.wm[index])/num_elem)
    print("WM voxels: "+str(num_elem))
    table.write("WM_perfusion "+mean+" "+str(wm_volume)+"\n")

    index=seg_csf>0#retrieve the pixel indexes belonging to cerebrospinal fluid
    num_elem=seg_csf[index].sum()
    print("CSF voxels: "+str(num_elem))
    mean=str(np.sum(perfusion.csf[index])/num_elem)
    table.write("CSF_perfusion "+mean+" "+str(csf_volume)+"\n\n")

    if (self.enableLabel.isChecked() and self.labelPath) or self.applyNorm.isChecked():
       c=0
       table.write("LABEL_ID REGION_NAME TOTAL_MEAN GM_MEAN WM_MEAN CSF_MEAN\n")
       for label in labelValues:
          table.write(str(label)+" "+labelRegions[c]+" ")
          label_mask=label_img==label
          num_elem=label_mask.sum()##for computing the mean perfusion value of the region count the elements on the normalized label volume if normalization was performed as perfusion.total is the normalized perfusion image
          if index.any() and num_elem:
            mean=str(np.sum(perfusion.total[label_mask])/num_elem)
            table.write(mean+" ")
          else:
            table.write("0 ")

          gm_vox=seg_gm*label_mask#select the voxels belonging to both the current region and grey matter
          index=gm_vox>0##retrieve the indexes of the voxels belonging to both the current region and grey matter 
          num_elem=gm_vox[index].sum()#obtain the number of voxels belonging to both the current region and grey matter
          if index.any() and num_elem:
            mean=str(np.sum(perfusion.gm[index])/num_elem)
            table.write(mean+" ")
          else:
            table.write("0 ")

          wm_vox=seg_wm*label_mask#select the voxels belonging to both the current region and white matter
          index=wm_vox>0##retrieve the indexes of the voxels belonging to both the current region and white matter
          num_elem=wm_vox[index].sum()#obtain the number of voxels belonging to both the current region and white matter
          if index.any() and num_elem:
            mean=str(np.sum(perfusion.wm[index])/num_elem)
            table.write(mean+" ")
          else:
            table.write("0 ")

          csf_vox=seg_csf*label_mask#select the voxels belonging to both the current region and cerebrospinal fluid
          index=csf_vox>0##retrieve the indexes of the voxels belonging to both the current region and cerebrospinal fluid
          num_elem=csf_vox[index].sum()#obtain the number of voxels belonging to both the current region and cerebrospinal fluid
          if index.any() and num_elem:
            mean=str(np.sum(perfusion.csf[index])/num_elem)
            table.write(mean)
          else:
            table.write("0")

          table.write("\n")
          c=c+1

    table.write("\nJob_start "+jobtime+"\n")
    jobtime=datetime.datetime.now().isoformat()
    pos=jobtime.rfind('.')
    jobtime=jobtime[0:pos]
    table.write("Job_finish "+jobtime+"\n")
    table.close()

class PVC_vFinalLogic(ScriptedLoadableModuleLogic):

  #3DWLS ALGORITHM

  def WLS3D(self,seg_gm,seg_wm,seg_csf,CBF,pixel_size,kernel_size,weight_type):

    def get_weight_kernel(kernel_size,pixel_size,weight_type):
      kernel_size=np.array(kernel_size)
      pixel_size=np.array(pixel_size)
      for i in range(0,3):
        if not(kernel_size[i] % 2):
          kernel_size[i]=kernel_size[i]+1;
      aux_x=np.zeros(kernel_size)##column coordinates in pixels
      aux_y=np.zeros(kernel_size)##row coordinates in pixels
      aux_z=np.zeros(kernel_size)##slice coordinates in pixels
      base=np.ones([kernel_size[1],kernel_size[2]])
      z=np.absolute(np.mgrid[-kernel_size[0]/2+1:kernel_size[0]/2+1])
      x=np.absolute(np.mgrid[-kernel_size[1]/2+1:kernel_size[1]/2+1])
      y=np.absolute(np.mgrid[-kernel_size[2]/2+1:kernel_size[2]/2+1])
      for i in range(0,kernel_size[0]):
         aux_z[i,:,:]=base*z[i]
      for i in range(0,kernel_size[1]):
         aux_x[:,i,:]=x
      for i in range(0,kernel_size[2]):
         aux_y[:,:,i]=y
      aux_z=aux_z*pixel_size[0]
      aux_x=aux_x*pixel_size[1]
      aux_y=aux_y*pixel_size[2]
      D=np.sqrt(np.power(aux_x,2)+np.power(aux_y,2)+np.power(aux_z,2))
      ##Each voxel of D matrix contains the euclidean distance of that voxel to the central voxel of the matrix.
      ##This distance is the real distance taken from the pixel_size values
      center=kernel_size/2;
      if weight_type.lower()=="distance":
          D[center[0],center[1],center[2]]=1;
          WE=1/D;##the weighting kernel in this case will be the inverse of the euclidean distance matrix D
      elif weight_type.lower()=="exponential":
          WE=np.exp(-D)##the weighting kernel in this case will be the inverse exponential of the euclidean distance matrix D
          WE[center[0],center[1],center[2]]=1;
      elif weight_type.lower()=="gaussian":
          #the weighting kernel in this case will be the gaussian of the distance values according to the gaussian equation:
          #WE(k,i,j))=A*exp{- [(D(k,i,j)-mean)^2] / [2*sigma^2]  }
          #mean=0 for the gaussian bell to be centered at zero i.e. to start the decay at the central position of the kernel D(center)=0
          #A coefficient determines the height of the gaussian bell; being the value attained at the center equal to A
          #sigma coefficient determines the width of the gaussian bell, i.e. the steepness of its decay from the central initial value A
          val=0.67#the value "val" attained at distance "dist" can be controlled by giving sigma a value according to the next formula:
          dist=np.min(pixel_size)##i.e set dist to be the closest voxel or distance to the center voxel, so that voxel will have a value=val
          sigma=dist/np.sqrt(-2*np.log(val))
          A=1##determina el alto de la gausiana e.g.a=1 el valor en el centro sera 1 y decaera mas o menos dependiendo del valor de sigma
          WE=A*np.exp(-(D**2)/(2*(sigma**2)))
      return WE

    def zeroPadd(matrix,upRows,downRows,leftCols,rightCols,zSliceFront,zSliceBack):
      matrix=np.array(matrix);
      [p,r,c]=matrix.shape
      out=np.zeros([p+zSliceFront+zSliceBack,r+upRows+downRows,c+leftCols+rightCols])
      out[zSliceFront:zSliceFront+p,upRows:r+upRows,leftCols:c+leftCols]=matrix;
      return out;

    def lscov(A,b,w):
      A=np.array(A)
      b=np.array(b)
      w=np.array(w)
      l=len(w)
      d=np.zeros([l,l])
      diag_ind=range(l)
      d[diag_ind,diag_ind]=w
      out=np.dot(A.T,d)
      out=np.dot(out,A)
      out=np.linalg.inv(out)
      out=np.dot(out,A.T)
      out=np.dot(out,d)
      out=np.dot(out,b)
      return out;

    class PerfusionSet:

      def __init__(self,gm,wm,csf):
          self.gm = gm
          self.wm = wm
          self.csf= csf
          self.total = gm+wm+csf

    seg_gm=np.array(seg_gm)
    seg_wm=np.array(seg_wm)
    seg_csf=np.array(seg_csf)
    CBF=np.array(CBF)
    seg_gm.astype(float)
    seg_wm.astype(float)
    seg_csf.astype(float)
    CBF.astype(float)
    kernel_size=np.array(kernel_size);
    pixel_size=np.array(pixel_size);
    if not kernel_size.shape:
        kernel_size=kernel_size.astype(float);#casting on numpy arrays
        kernel_size=kernel_size/pixel_size;
        kernel_size=kernel_size.round();
    prod=1;
    kernel_size=kernel_size.astype(int);
    for i in range(3):
        if (not(kernel_size[i] % 2) or not(kernel_size[i])):
            kernel_size[i]=kernel_size[i]+1;
        prod=prod*kernel_size[i];
    if prod==1:
        kernel_size[0]=3;
        kernel_size[1]=3;

    #print("\nKernel dimensions (pixels): %d x %d x %d" % (kernel_size[0],kernel_size[1],kernel_size[2]))
    #print("Kernel dimensions (mm): %f x %f x %f\n" % (kernel_size[0]*pixel_size[0],kernel_size[1]*pixel_size[1],kernel_size[2]*pixel_size[2]))
    #print("Weighting matrix type: "+weight_type+"\n")

    kernel_dims=[kernel_size[2],kernel_size[0],kernel_size[1]]
    pixel_dims=[pixel_size[2],pixel_size[0],pixel_size[1]]
    W=get_weight_kernel(kernel_dims,pixel_dims,weight_type)##Weighting matrix
    dims=W.shape
    n_elem=np.prod(dims)
    W=W.reshape(n_elem)##convert the weighting matrix into a vector
    p_extend=kernel_size[2]/2
    r_extend=kernel_size[0]/2
    c_extend=kernel_size[1]/2
    aux_GM=seg_gm
    aux_WM=seg_wm
    aux_CSF=seg_csf
    [p,r,c]=seg_gm.shape
    CBF_GM=np.zeros([p,r,c])
    CBF_WM=np.zeros([p,r,c])
    CBF_CSF=np.zeros([p,r,c])
    brain_mask=seg_gm+seg_wm+seg_csf
    brain_mask[brain_mask>0]=1
    CBF=CBF*brain_mask#preprocessing of the perfusion image-> only voxels belonging to brain tissue are kept for analysis, others are set to zero.
    seg_gm=zeroPadd(seg_gm,r_extend,r_extend,c_extend,c_extend,p_extend,p_extend)
    seg_wm=zeroPadd(seg_wm,r_extend,r_extend,c_extend,c_extend,p_extend,p_extend)
    seg_csf=zeroPadd(seg_csf,r_extend,r_extend,c_extend,c_extend,p_extend,p_extend)
    CBF=zeroPadd(CBF,r_extend,r_extend,c_extend,c_extend,p_extend,p_extend)
    maximum=CBF.max()
    progress=0
    def_count=0
    for k in range(p_extend,p+p_extend):
        progress+=1
        #print("Processing slice %d out of %d" % (progress,p))
        ip=[k-p_extend,k+p_extend+1]
        for i in range(r_extend,r+r_extend):
            ir=[i-r_extend,i+r_extend+1]
            for j in range(c_extend,c+c_extend):
                ic=[j-c_extend,j+c_extend+1]
                ##retrieve the tissue probabilities of the current central voxel and the perfusion value
                vox_pgm=seg_gm[k,i,j]
                vox_pwm=seg_wm[k,i,j]
                vox_pcsf=seg_csf[k,i,j]
                vox_cbf=CBF[k,i,j]
                if vox_cbf==0:##if the current central voxel has no perfusion, the individual perfusion values must be also zero so move to the next loop iteration
                  continue
                elif vox_pgm==1 or (vox_pwm==0 and vox_pcsf==0):##if the central voxel is made exclusively of one tissue, then the perfusion must result only for that tissue type
                  CBF_GM[ip[0],ir[0],ic[0]]=CBF[k,i,j]
                elif vox_pwm==1 or (vox_pgm==0 and vox_pcsf==0):
                  CBF_WM[ip[0],ir[0],ic[0]]=CBF[k,i,j]
                elif vox_pcsf==1 or (vox_pgm==0 and vox_pwm==0):
                  CBF_CSF[ip[0],ir[0],ic[0]]=CBF[k,i,j]
                else: ##if not, apply 3dwls regression algorithm
                  b=CBF[ip[0]:ip[1],ir[0]:ir[1],ic[0]:ic[1]]#retrive the perfusion values of the current neighborhood
                  p_GM=seg_gm[ip[0]:ip[1],ir[0]:ir[1],ic[0]:ic[1]]#retrive the tissue probabilities of the current neighborhood
                  p_WM=seg_wm[ip[0]:ip[1],ir[0]:ir[1],ic[0]:ic[1]]
                  p_CSF=seg_csf[ip[0]:ip[1],ir[0]:ir[1],ic[0]:ic[1]]
                  condition=p_GM+p_WM+p_CSF
                  condition[condition>0]=1
                  condition=np.sum(condition)
                  if condition>3:##if more than three voxels are available (i.e. more than three voxels have some amount of tissue assigned -not background)
                      p_GM=p_GM.reshape([n_elem,1])##convert the matrix containing the tissue probability maps of the current neighborhood into column vectors
                      p_WM=p_WM.reshape([n_elem,1])
                      p_CSF=p_CSF.reshape([n_elem,1])
                      b=b.reshape(n_elem)
                      A=np.concatenate([p_GM,p_WM,p_CSF],1)##concatenate in the column direction
                      idx=A>0
                      idx=idx.any(axis=0)##check for columns whose row elements are all higher than zero
                      A=A[:,idx]#keep only those columns of the matrix to avoid rank deficiency -and matrix singularity- when solving least squares
                      x=np.zeros(3)
                      try:##in case the resulting matrix is still singular, a try statement is used. If lscov fails, the voxel will be assigned zero intensity value.
                        s=lscov(A,b,W)
                        x[idx]=s
                        CBF_GM[ip[0],ir[0],ic[0]]=x[0]
                        CBF_WM[ip[0],ir[0],ic[0]]=x[1]
                        CBF_CSF[ip[0],ir[0],ic[0]]=x[2]
                      except:
                        def_count=def_count+1
##                      if x[0]>5000 or x[0]<-300:
##                        print(A)
##                        print(b)
##                        print(CBF[k,i,j])
##                        print("GM Intensity error at "+str(ip[0])+" "+str(ir[0])+" "+str(ic[0])+"//value="+str(x[0]))
##                        print("Tissue probabilities: "+str(vox_pgm)+" "+str(vox_pwm)+" "+str(vox_pcsf))
##                      if x[1]>5000 or x[1]<-300:
##                        print(A)
##                        print(b)
##                        print(CBF[k,i,j])
##                        print("WM Intensity error at "+str(ip[0])+" "+str(ir[0])+" "+str(ic[0])+"//value="+str(x[1]))
##                        print("Tissue probabilities: "+str(vox_pgm)+" "+str(vox_pwm)+" "+str(vox_pcsf))
##                      if x[2]>5000 or x[2]<-300:
##                        print(A)
##                        print(b)
##                        print(CBF[k,i,j])
##                        print("CSF Intensity error at "+str(ip[0])+" "+str(ir[0])+" "+str(ic[0])+"//value="+str(x[2]))
##                        print("Tissue probabilities: "+str(vox_pgm)+" "+str(vox_pwm)+" "+str(vox_pcsf))          
    #print("Total count of singular matrix errors:   "+str(def_count))
    gm=CBF_GM*aux_GM
    wm=CBF_WM*aux_WM
    csf=CBF_CSF*aux_CSF                    
    #set ocurrences of negative perfusion values to zero, set any value surpassing the maximum value of the original CBF to that value
    CBF_GM[CBF_GM>maximum]=maximum
    CBF_WM[CBF_WM>maximum]=maximum
    CBF_CSF[CBF_CSF>maximum]=maximum 
    CBF_GM[CBF_GM<0]=0
    CBF_WM[CBF_WM<0]=0
    CBF_CSF[CBF_CSF<0]=0

    perfusion=PerfusionSet(gm,wm,csf)
    #print('DONE!')
    return perfusion;

