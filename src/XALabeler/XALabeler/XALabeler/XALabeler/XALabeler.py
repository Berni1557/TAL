import logging
import os
import sys
import vtk
import time
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import SegmentStatistics
from settings.settings import Settings
from settings.settingsPath import settingsPath
from ALAction.ALAction import ALAction
import qt
import json
import numpy as np 
import SimpleITK as sitk
                    
def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)
        
#
# XALabeler
#

class XALabeler(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "XALabeler"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["XAL"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#XALabeler">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # XALabeler1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='XALabeler',
        sampleName='XALabeler1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'XALabeler1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='XALabeler1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='XALabeler1'
    )

    # XALabeler2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='XALabeler',
        sampleName='XALabeler2',
        thumbnailFileName=os.path.join(iconsPath, 'XALabeler2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='XALabeler2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='XALabeler2'
    )


#
# XALabelerWidget
#

class XALabelerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.actionlist = []
        self.actionIdx = -1
        self.volumeNode = None
        self.pseudoNode = None
        self.maskNode = None
        self.roiNode = []
        self.mip = False
        
    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/XALabeler.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = XALabelerLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Setup SegmentEditorWidget
        self.ui.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.ui.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
        self.ui.segmentEditorWidget.setEffectNameOrder(["Paint", "Erase"])
        self.ui.segmentEditorWidget.unorderedEffectsVisible = False
        self.ui.segmentEditorWidget.setActiveEffectByName("Paint")
        self.ui.segmentEditorWidget.show()

        # Update layoutManager
        self.layoutManager = slicer.app.layoutManager()
        self.layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
        self.red = self.layoutManager.sliceWidget('Red')
        self.redLogic = self.red.sliceLogic()
        
        # Read settings
        self.settings = Settings()
        #fip_settings = os.path.dirname(os.path.dirname(
        #    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/XALabeler/data_manual/settings_XALabeler.json'
        fip_settings = settingsPath
        if not os.path.exists(fip_settings):
            self.settings.writeSettings(fip_settings)
        self.settings.readSettings(fip_settings)

    	#!!! Update settings
        #self.settings['fip_actionlist'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/XALabeler/XALabeler/data_manual/actionlist.json'
        #self.settings['fip_actionlist'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XALabeler/XALabeler/data_manual/actionlist.json'
        # Replace filepath
        # old_string = '/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src'
        # new_string = '/mnt/SSD2/cloud_data/Projects/CTP/src'
        # inplace_change(self.settings['fip_actionlist'], old_string, new_string)
        
        # Update label
        self.ui.label.setText('Welcome to XALabeler!')
        
        # Buttons
        #self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.startButton.connect('clicked(bool)', self.onStartButton)
        self.ui.nextButton.connect('clicked(bool)', self.onNextButton)
        self.ui.backButton.connect('clicked(bool)', self.onBackButton)
        self.ui.stopButton.connect('clicked(bool)', self.onStopButton)
        
        # Progress bar
        self.ui.progressBar.setValue(0)
        
        # Extract color labels
        #colorNode = slicer.util.loadColorTable('/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/XALabeler/XALabeler/data_manual/XALabelerLUT.ctbl')
        #colorNode = slicer.util.loadColorTable('/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XALabeler/XALabeler/data_manual/XALabelerLUT.ctbl')
        colorNode = slicer.util.loadColorTable(self.settings['fip_colors'])
        for i in range(colorNode.GetNumberOfColors()):
            cname = colorNode.GetColorName(i)
            colorNode.SetColorName(i, cname.replace(' ', '_'))
        self.colorLabels = [colorNode.GetColorName(i) for i in range(colorNode.GetNumberOfColors())]

        # Add shortcuts
        shortcuts = [
          ('n', lambda: self.onNextButton()),
          ('b', lambda: self.onBackButton()),
          ('q', lambda: self.onStopButton()),
          ('t', lambda: self.onToggleVisibility()),
          ('r', lambda: self.onToggleROIVisibility()),
          ('1', lambda: self.onNumberButton(1)),
          ('2', lambda: self.onNumberButton(2)),
          ('3', lambda: self.onNumberButton(3)),
          ('4', lambda: self.onNumberButton(4)),
          ('5', lambda: self.onNumberButton(5)),
          ('6', lambda: self.onNumberButton(6)),
          ('7', lambda: self.onNumberButton(7)),
          ('8', lambda: self.onNumberButton(8)),
          ('9', lambda: self.onNumberButton(9)),
          ('Ctrl+,', lambda: slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)),
          ('m', lambda: self.onMIP())
          ]
        
        for (shortcutKey, callback) in shortcuts:
            shortcut = qt.QShortcut(slicer.util.mainWindow())
            shortcut.setKey(qt.QKeySequence(shortcutKey))
            shortcut.connect( 'activated()', callback)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        #self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        #self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
        #self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
        #self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        #self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

        # # Update buttons states and tooltips
        # if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
        #     self.ui.applyButton.toolTip = "Compute output volume"
        #     self.ui.applyButton.enabled = True
        # else:
        #     self.ui.applyButton.toolTip = "Select input and output volume nodes"
        #     self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onStartButton(self):
        """
        Run processing when user clicks "Strat" button.
        """
        print('onStartButton123')
        self.ui.startButton.setEnabled(False)

        # Load actionlist
        self.actionlist = ALAction.load(self.settings['fip_actionlist'])
        
        # Init actionIdx
        self.actionIdx = len(self.actionlist)
        for i, action in enumerate(self.actionlist):
            if action.status == 'open':
                self.actionIdx = i
                break
            
        # Update progressBar
        self.ui.progressBar.setValue((self.actionIdx/len(self.actionlist))*100)
        
        # Update instructions
        if self.settings['classification']:
            self.ui.label.setText('Classiefie the image 1-positive, 2-negative.')
        else:
            self.ui.label.setText('Segment the image.')
            
        # Init actionIdxSave
        self.actionIdxSave=[]

        # Load first slice
        if len(self.actionlist) > 0 and self.actionIdx<len(self.actionlist):
            self.process()
            
        
    def onNextButton(self):
        """
        Next
        """
        
        print('self.colorLabels02', self.colorLabels)

        
        #print('onNextButton123')
        #print('actionIdx01', self.actionIdx)
        if not self.actionlist[self.actionIdx].info['classified']:
            self.actionlist[self.actionIdx].info['classified']=True
            #self.actionlist[self.actionIdx].info['class']=-1

        # Update actionIdx
        if not self.settings['classification_multislice']:
            self.actionIdxSave = [self.actionIdx]
            if self.actionIdx < len(self.actionlist):
                self.actionIdx = self.actionIdx + 1
        else:
            self.actionIdxSave = []
            #for i in range(self.actionIdx+1, len(self.actionlist)):
            for i in range(self.actionIdx+1, len(self.actionlist)):
                if self.actionlist[i].imagename != self.actionlist[self.actionIdx].imagename:
                    self.actionIdx=i
                    self.actionIdxSave.append(i-1)
                    break
                else:
                    self.actionlist[i].info['classified']=True
                self.actionIdxSave.append(i-1)
        #print('actionIdx02', self.actionIdx)
            
        self.process()
        
    # def onNextButton(self):
    #     """
    #     Next
    #     """
    #     print('onNextButton123')
    #     if not self.actionlist[self.actionIdx].info['classified']:
    #         self.actionlist[self.actionIdx].info['classified']=True
    #         #self.actionlist[self.actionIdx].info['class']=-1

    #     # Update actionIdx
    #     if not self.settings['classification_multislice']:
    #         self.actionIdxSave = [self.actionIdx]
    #         if self.actionIdx < len(self.actionlist):
    #             self.actionIdx = self.actionIdx + 1
    #     else:
    #         self.actionIdxSave = []
    #         for i in range(self.actionIdx+1, len(self.actionlist)):
    #             if self.actionlist[i].imagename != self.actionlist[self.actionIdx].imagename:
    #                 self.actionIdx=i
    #                 self.actionIdxSave.append(i-1)
    #                 break
    #             else:
    #                 self.actionlist[i].info['classified']=True
    #             self.actionIdxSave.append(i-1)
            
    #     self.process()

    def onBackButton(self):
        """
        Back
        """

        #print('onBackButton123')
        
        #print('actionIdx03', self.actionIdx)
        
        if self.actionlist[self.actionIdx].info['classified']:
            self.actionlist[self.actionIdx].info['classified']=False
           
        if not self.settings['classification_multislice']:
            #self.actionIdxSave = [self.actionIdx]
            if self.actionIdx > 0:
                self.actionIdx = self.actionIdx - 1
                self.actionlist[self.actionIdx].status = 'open'
                self.actionIdxSave = []

        # Update actionIdx
        #if self.actionIdx > 0:
        #    self.actionIdx = self.actionIdx - 1
        #    self.actionlist[self.actionIdx].status = 'open'
        
        #print('actionIdx04', self.actionIdx)
            
        self.process()
        
    # def onBackButton(self):
    #     """
    #     Back
    #     """

    #     # Update actionIdx
    #     if self.actionIdx > 0:
    #         self.actionIdx = self.actionIdx - 1
    #         self.actionlist[self.actionIdx].status = 'open'
    #     self.process()

    def onNumberButton(self, number):
        if self.settings['classification']:
            #self.onClassButton(number)
            if self.settings['classification_multislice']:
                #if not 'class' in self.actionlist[self.actionIdx].info:
                #    self.actionlist[self.actionIdx].info['class']=[]
                self.actionlist[self.actionIdx].info['classified']=True
                sl = int((self.redLogic.GetSliceOffset()-self.volumeNode.GetOrigin()[2])/self.volumeNode.GetSpacing()[2])
                self.actionlist[self.actionIdx].info['class'].append((sl, int(number)))
                #self.onNextButton()                    
            else:
                self.actionlist[self.actionIdx].info['classified']=True
                #self.actionlist[self.actionIdx].info['class']=int(number)
                sl = int((self.redLogic.GetSliceOffset()-self.volumeNode.GetOrigin()[2])/self.volumeNode.GetSpacing()[2])
                self.actionlist[self.actionIdx].info['class'].append((sl, int(number)))
                self.onNextButton()
        else:
            if number>0 and number<=len(self.colorLabels):
                self.segmentEditorNode.SetSelectedSegmentID(self.colorLabels[number-1])
                
    def onMIP(self):
        print('onMIP', self.mip)
        if self.mip:
            sliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
            appLogic = slicer.app.applicationLogic()
            sliceLogic = appLogic.GetSliceLogic(sliceNode)
            sliceLayerLogic = sliceLogic.GetBackgroundLayer()
            reslice = sliceLayerLogic.GetReslice()
            reslice.SetSlabModeToMax()
            reslice.SetSlabNumberOfSlices(20) # mean of 10 slices will computed
            reslice.SetSlabSliceSpacingFraction(0.3) # spacing between each slice is 0.3 pixel (total 10 * 0.3 = 3 pixel neighborhood)
            sliceNode.Modified()
            self.mip=False
        else:
            sliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
            appLogic = slicer.app.applicationLogic()
            sliceLogic = appLogic.GetSliceLogic(sliceNode)
            sliceLayerLogic = sliceLogic.GetBackgroundLayer()
            reslice = sliceLayerLogic.GetReslice()
            reslice.SetSlabModeToSum()
            reslice.SetSlabNumberOfSlices(1) # mean of 10 slices will computed
            reslice.SetSlabSliceSpacingFraction(0.3) # spacing between each slice is 0.3 pixel (total 10 * 0.3 = 3 pixel neighborhood)
            sliceNode.Modified()
            self.mip=True
            
            
    def onToggleVisibility(self):
        segmentationNode=slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        segmentationDisplayNode=segmentationNode.GetDisplayNode()
        vis = segmentationDisplayNode.GetVisibility()
        segmentationDisplayNode.SetVisibility(not vis)

    def onToggleROIVisibility(self):
        roiNodes=slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiNodes:
            roiDisplayNode=roiNode.GetDisplayNode()
            vis = roiDisplayNode.GetVisibility()
            roiDisplayNode.SetVisibility(not vis)
        
        # roiNode=slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsROINode")
        # roiDisplayNode=roiNode.GetDisplayNode()
        # vis = roiDisplayNode.GetVisibility()
        # roiDisplayNode.SetVisibility(not vis)
        
    # def onClassButton(self, number):
    #     print('onClassButton')
    #     #actionIdxSave = self.actionIdx - 1
    #     self.actionlist[self.actionIdx].info['classified']=True
    #     self.actionlist[self.actionIdx].info['class']=int(number)
    #     #action = self.actionlist[self.actionIdx-1].msg=number
    #     #if number>0 and number<=len(self.colorLabels):
    #     #    self.segmentEditorNode.SetSelectedSegmentID(self.colorLabels[number-1])
    def setWindowLevel(self, window=800, level=250):
        self.volumeNode.GetDisplayNode().SetAutoWindowLevel(False)
        self.volumeNode.GetDisplayNode().SetWindowLevel(window, 250)
            
    def onStopButton(self):
        """
        Stop
        """
        self.ui.startButton.setEnabled(True)
        self.ui.progressBar.setValue(0)
        slicer.mrmlScene.Clear()
        pass

    def reslice_func(self, volumeNode, resolution=0.25, name='resliced'):
        
        global res
        
        def waitUntilReslice(timeout, res, period=0.25):
              mustend = time.time() + timeout
              while time.time() < mustend:
                if (res.GetStatusString() == 'Completed') or \
                    (res.GetStatusString() == 'Completing'): # I want to use only "Completed"
                    print(res.GetStatusString())
                    return True
                time.sleep(period)
              return False
    
        #reslicedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        parameters = {
            "outputPixelSpacing":"{:},{:},{:}".format(*[resolution]*3),
            "InputVolume":volumeNode.GetID(),
            "interpolationMode":'bspline',
            "referenceVolume": volumeNode.GetID(),
            "OutputVolume":volumeNode.GetID()}
        
        res = slicer.cli.run(slicer.modules.resamplescalarvolume, None, parameters)
        
        waitUntilReslice(60, res)
        
        # reslicedVolumeNode = slicer.util.getNode(name)
        
        if(not volumeNode == None):
               volumeNode.GetDisplayNode().SetAutoWindowLevel(0)
               volumeNode.GetDisplayNode().SetWindowLevel(800,100)
    
        return volumeNode

    def process(self):
        import time
        start = time.time()
        
        #actionIdxSave = self.actionIdx - 1
        # Save segmentation if exist
        if len(self.actionIdxSave) > 0:
            action = self.actionlist[self.actionIdxSave[0]]
            if self.pseudoNode is not None:
                if action.fip_refine is None:
                    fip_refine = os.path.join(self.settings['fp_refine'], action.refinename)
                else:
                    fip_refine = action.fip_refine

                labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
                
                ###
                #arr = slicer.util.arrayFromVolume(labelmapVolumeNode)
                #arr = arr + 100
                #slicer.util.updateVolumeFromArray(labelmapVolumeNode, arr)
                
                segmentIds = self.pseudoNode.GetSegmentation().GetSegmentIDs()
                slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(self.pseudoNode, segmentIds, labelmapVolumeNode, self.volumeNode)
                #slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(self.pseudoNode, labelmapVolumeNode)
                
                # Check if pseudo label was used
                if action.fip_pseudo is None:
                    fip_pseudo = os.path.join(self.settings['fp_pseudo'], action.pseudoname)
                else:
                    fip_pseudo = action.fip_pseudo
                pseudo_used = os.path.isfile(fip_pseudo)
                    
                # Correct the offset that backround has label 0
                #if self.settings['ignore_default']:
                #if True:
                #print('default_ignore123', self.settings['default_ignore'])
                #print('colorLabels123', self.colorLabels)
                #label_ignore = len(self.colorLabels)-1
                label_ignore = self.colorLabels.index('ignore')
                label_pseudo = self.colorLabels.index('pseudo')
                print('label_ignore123', label_ignore)
                if self.settings['default_ignore'] and not pseudo_used:
                    labelmapArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
                    labelmapArray = labelmapArray-1
                    labelmapArray[labelmapArray==-1]=label_ignore
                    labelmapVolumeNodeMod = slicer.util.addVolumeFromArray(labelmapArray)
                    labelmapVolumeNodeMod.SetSpacing(labelmapVolumeNode.GetSpacing())
                    labelmapVolumeNodeMod.SetOrigin(labelmapVolumeNode.GetOrigin())
                    #labelmapVolumeNodeMod.SetDirection(labelmapVolumeNode.GetDirection())
                    ijkdirs = [[0,0,0],[0,0,0],[0,0,0]]
                    labelmapVolumeNode.GetIJKToRASDirections(ijkdirs)
                    labelmapVolumeNodeMod.SetIJKToRASDirections(ijkdirs)
                    slicer.util.saveNode(labelmapVolumeNodeMod, fip_refine)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNodeMod)
                    
                elif self.settings['default_ignore'] and pseudo_used:
                    # pseudo_im = sitk.ReadImage(fip_pseudo)
                    # pseudoArray = sitk.GetArrayFromImage(pseudo_im)
                    # labelmapArray = slicer.util.arrayFromVolume(labelmapVolumeNode).astype(np.int16)
                    # print('labelmapArray123max', labelmapArray.max())
                    # labelmapArrayTmp = labelmapArray-1
                    # labelmapArrayTmp[labelmapArrayTmp==-1]=0
                    # label_diff = ((labelmapArrayTmp!=pseudoArray)*1).astype(np.int16)
                    # labelmapArray = (labelmapArray*label_diff).astype(np.int16)
                    # labelmapArray = labelmapArray-1
                    # labelmapArray[labelmapArray==-1]=label_ignore
                    # labelmapVolumeNodeMod = slicer.util.addVolumeFromArray(labelmapArray)
                    # labelmapVolumeNodeMod.SetSpacing(labelmapVolumeNode.GetSpacing())
                    # labelmapVolumeNodeMod.SetOrigin(labelmapVolumeNode.GetOrigin())
                    # ijkdirs = [[0,0,0],[0,0,0],[0,0,0]]
                    # labelmapVolumeNode.GetIJKToRASDirections(ijkdirs)
                    # labelmapVolumeNodeMod.SetIJKToRASDirections(ijkdirs)
                    # slicer.util.saveNode(labelmapVolumeNodeMod, fip_refine)
                    # slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
                    # slicer.mrmlScene.RemoveNode(labelmapVolumeNodeMod)
                    
                    
                    print('label_pseudo123', label_pseudo)
                    pseudo_im = sitk.ReadImage(fip_pseudo)
                    pseudoArray = sitk.GetArrayFromImage(pseudo_im)
                    labelmapArray = slicer.util.arrayFromVolume(labelmapVolumeNode).astype(np.int16)
                    print('labelmapArray123', np.unique(labelmapArray, return_counts=True))
                    print('pseudoArray123', np.unique(pseudoArray, return_counts=True))
                    print('labelmapArray1234', labelmapArray.shape)
                    print('pseudoArray1234', pseudoArray.shape)
                    idx_bg = np.where(labelmapArray==1) 
                    diff = np.zeros(labelmapArray.shape)
                    for lab in range(1,len(segmentIds)):
                        print('lab123', lab)
                        diff = diff + ((labelmapArray==lab+1) != (pseudoArray==lab))
                    diff = diff>0
                    print('diff123', diff.sum())
                    #sys.exit()
                        
                    idx_class_changed = np.where(diff==True)
                    idx_pseudo = np.where(labelmapArray==label_pseudo+1)
                    #idx_ignore = np.where(labelmapArray==label_ignore)
                    labelmapOut = np.ones(labelmapArray.shape)*label_ignore
                    print('idx_changed123', idx_class_changed[0].shape)
                    print('idx_pseudo', idx_pseudo[0].shape)
                    labelmapOut[idx_class_changed] = labelmapArray[idx_class_changed]-1
                    labelmapOut[idx_pseudo] = pseudoArray[idx_pseudo]
                    labelmapOut[idx_bg] = 0

                    
                    #print('labelmapArray123max', labelmapArray.max())
                    #labelmapArrayTmp = labelmapArray-1
                    #labelmapArrayTmp[labelmapArrayTmp==-1]=label_ignore
                    #labelmapArrayTmp = labelmapArray
                    #label_diff = ((labelmapArrayTmp!=pseudoArray)*1).astype(np.int16)
                    #labelmapArray = (labelmapArray*label_diff).astype(np.int16)
                    #label_diff = ((labelmapArrayTmp>0)*1).astype(np.int16)
                    #arr_pseudo = ((labelmapArrayTmp==label_pseudo)*1).astype(np.int16)
                    #labelmapArray = (1-arr_pseudo)*labelmapArray*label_diff+arr_pseudo*pseudoArray
                    
                    #labelmapArray = labelmapArray-1
                    #labelmapArray[labelmapArray==-1]=label_ignore
                    
                    #labelmapVolumeNodeMod = slicer.util.addVolumeFromArray(labelmapArray)
                    
                    labelmapVolumeNodeMod = slicer.util.addVolumeFromArray(labelmapOut)
                    labelmapVolumeNodeMod.SetSpacing(labelmapVolumeNode.GetSpacing())
                    labelmapVolumeNodeMod.SetOrigin(labelmapVolumeNode.GetOrigin())
                    ijkdirs = [[0,0,0],[0,0,0],[0,0,0]]
                    labelmapVolumeNode.GetIJKToRASDirections(ijkdirs)
                    labelmapVolumeNodeMod.SetIJKToRASDirections(ijkdirs)
                    slicer.util.saveNode(labelmapVolumeNodeMod, fip_refine)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNodeMod)
                    
                else:
                    labelmapArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
                    labelmapArray = labelmapArray-1
                    labelmapArray[labelmapArray==-1]=0
                    labelmapVolumeNodeMod = slicer.util.addVolumeFromArray(labelmapArray)
                    labelmapVolumeNodeMod.SetSpacing(labelmapVolumeNode.GetSpacing())
                    labelmapVolumeNodeMod.SetOrigin(labelmapVolumeNode.GetOrigin())
                    labelmapVolumeNodeMod.SetDirection(labelmapVolumeNode.GetDirection())
                    slicer.util.saveNode(labelmapVolumeNodeMod, fip_refine)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
                    slicer.mrmlScene.RemoveNode(labelmapVolumeNodeMod)
        
        print('Endtime0', time.time() - start)
        
        # Save actionlist
        if self.actionIdx > 0:
            for idxSave in self.actionIdxSave:
                self.actionlist[idxSave].status = 'solved'
                if self.settings['classification']:
                    self.actionlist[idxSave].info['annotated']=False
                else:
                    self.actionlist[idxSave].info['annotated']=True
            ALAction.save(self.settings['fip_actionlist'], self.actionlist)

        # Extract next action
        idx_start = self.actionIdx
        for i in range(idx_start, len(self.actionlist)+1):
            self.actionIdx = i
            if self.actionIdx==len(self.actionlist):
                self.onStopButton()
                return
            action = self.actionlist[self.actionIdx]
            #print('action.status12', action.status)
            if action.status == 'open':
                break
    
        #print('time123')
        #time.sleep(100)
        
        # Update progressbar
        self.ui.progressBar.setValue((self.actionIdx/len(self.actionlist))*100)
        colorNode = slicer.util.loadColorTable(self.settings['fip_colors'])
        for i in range(colorNode.GetNumberOfColors()):
            cname = colorNode.GetColorName(i)
            colorNode.SetColorName(i, cname.replace(' ', '_'))
        
        # Update pseudoNode
        if slicer.mrmlScene.GetFirstNodeByName(action.pseudoname) is None:
            if self.pseudoNode is None:
                if action.fip_pseudo is None:
                    fip_pseudo = os.path.join(self.settings['fp_pseudo'], action.pseudoname)
                else:
                    fip_pseudo = action.fip_pseudo
                if os.path.isfile(fip_pseudo):
                    self.pseudoNode = slicer.util.loadSegmentation(fip_pseudo, properties={'name': action.pseudoname, 'colorNodeID': colorNode.GetID(), 'show': True})
                else:
                    self.pseudoNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", action.pseudoname)
            else:
                if action.fip_pseudo is None:
                    fip_pseudo = os.path.join(self.settings['fp_pseudo'], action.pseudoname)
                else:
                    fip_pseudo = action.fip_pseudo
                slicer.mrmlScene.RemoveNode(self.pseudoNode)
                if os.path.isfile(fip_pseudo):
                    self.pseudoNode = slicer.util.loadSegmentation(fip_pseudo, properties={'fileType': 'SegmentationFile', 'name': action.pseudoname, 'colorNodeID': colorNode.GetID(), 'show': True})
                else:
                    self.pseudoNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", action.pseudoname)
            self.ui.segmentEditorWidget.setSegmentationNode(self.pseudoNode)

        # Update volumeNode   
        if slicer.mrmlScene.GetFirstNodeByName(action.imagename) is None:
            if self.volumeNode is None:
                if action.fip_image is None:
                    fip_image = os.path.join(self.settings['fp_images'], action.imagename)
                else:
                    fip_image = action.fip_image
                self.volumeNode = slicer.util.loadVolume(fip_image, properties={'name': action.imagename, 'show': True})
            else:
                if action.fip_image is None:
                    fip_image = os.path.join(self.settings['fp_images'], action.imagename)
                else:
                    fip_image = action.fip_image
                slicer.mrmlScene.RemoveNode(self.volumeNode)
                self.volumeNode = slicer.util.loadVolume(fip_image, properties={'name': action.imagename, 'show': True})
            self.setWindowLevel()
            self.ui.segmentEditorWidget.setSourceVolumeNode(self.volumeNode)
            # Reset ROI
            for node in self.roiNode:
                slicer.mrmlScene.RemoveNode(node)
            self.roiNode=[]
            
            #!!!
            #self.volumeNode = self.reslice_func(self.volumeNode, resolution=0.2, name='resliced')

        # Add labels
        for label in action.label:
            if self.pseudoNode is not None:
                segment = self.pseudoNode.GetSegmentation().GetSegment(str(label[1]))
            else:
                segment = None
            if segment is not None:
                segment.SetColor(label[2][0], label[2][1], label[2][2])
                segment.SetName(label[1])
            else:
                segmentId = self.pseudoNode.GetSegmentation().AddEmptySegment(str(label[1]))
                segment = self.pseudoNode.GetSegmentation().GetSegment(segmentId)
                segment.SetColor(label[2][0], label[2][1], label[2][2])
                segment.SetName(label[1])
                self.pseudoNode.GetSegmentation().SetSegmentIndex(segmentId, label[0])
    
        # Update layoutManager
        if self.settings['classification_multislice']:
            slice_vis = 0
        else:
            if action.slice is not None:
                slice_vis = action.slice
            else:
                slice_vis = int((action.bboxLbsOrg[0]+action.bboxUbsOrg[0])/2)

        #offset = self.redLogic.GetSliceOffset()
        origen = self.volumeNode.GetOrigin()
        spacing = self.volumeNode.GetSpacing()
        offset = origen[2] + slice_vis * spacing[2]
        self.redLogic.SetSliceOffset(offset)
        
        # Set ROI
        if self.settings['show_roi']:
            #print('self.roiNode1235678', self.roiNode)
            if not self.settings['classification_multislice']:
                #print('self.roiNode12356', self.roiNode)
                #if self.roiNode is None:
                #    self.roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
                if len(self.roiNode)==0:
                    self.roiNode.append(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode"))
                # Set ROI center
                if action.dim==2:
                    dx = int((action.bboxUbsOrg[2]+action.bboxLbsOrg[2])/2)
                    dy = int((action.bboxUbsOrg[1]+action.bboxLbsOrg[1])/2)
                    dz = int((action.bboxUbsOrg[0]+action.bboxLbsOrg[0])/2)
                    ijkToRas = vtk.vtkMatrix4x4()
                    self.volumeNode.GetIJKToRASMatrix(ijkToRas)
                    position_Ijk=[dx, dy, dz, 1]
                    position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                    self.roiNode[0].SetCenter(position_Ras[0:3])
                    # Set ROI size
                    position_Ijk=[action.bboxLbsOrg[1], action.bboxLbsOrg[2], action.bboxLbsOrg[0], 1]
                    bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                    position_Ijk=[action.bboxUbsOrg[1], action.bboxUbsOrg[2], action.bboxUbsOrg[0], 1]
                    bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                    self.roiNode[0].SetSizeWorld([bboxU[0]-bboxL[0], bboxU[1]-bboxL[1], bboxU[2]-bboxL[2]])
                else:
                    dx = int((action.bboxUbsOrg[2]+action.bboxLbsOrg[2])/2)
                    dy = int((action.bboxUbsOrg[1]+action.bboxLbsOrg[1])/2)
                    dz = int((action.bboxUbsOrg[0]+action.bboxLbsOrg[0])/2)
                    ijkToRas = vtk.vtkMatrix4x4()
                    self.volumeNode.GetIJKToRASMatrix(ijkToRas)
                    position_Ijk=[dx, dy, dz, 1]
                    position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                    self.roiNode[0].SetCenter(position_Ras[0:3])
                    # Set ROI size
                    position_Ijk=[action.bboxLbsOrg[1], action.bboxLbsOrg[2], action.bboxLbsOrg[0], 1]
                    bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                    position_Ijk=[action.bboxUbsOrg[1], action.bboxUbsOrg[2], action.bboxUbsOrg[0], 1]
                    bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                    self.roiNode[0].SetSizeWorld([bboxU[0]-bboxL[0], bboxU[1]-bboxL[1], bboxU[2]-bboxL[2]])
            else:
                print('self.roiNode123', self.roiNode)
                for ac in self.actionlist:
                    if ac.imagename==action.imagename:
                        #if self.roiNode is None:
                        #    self.roiNode=[]
                        self.roiNode.append(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode"))
                        
                        # Set ROI center
                        if ac.dim==2:
                            dx = int((ac.bboxUbsOrg[2]+ac.bboxLbsOrg[2])/2)
                            dy = int((ac.bboxUbsOrg[1]+ac.bboxLbsOrg[1])/2)
                            dz = int((ac.bboxUbsOrg[0]+ac.bboxLbsOrg[0])/2)
                            ijkToRas = vtk.vtkMatrix4x4()
                            self.volumeNode.GetIJKToRASMatrix(ijkToRas)
                            position_Ijk=[dx, dy, dz, 1]
                            position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                            self.roiNode[-1].SetCenter(position_Ras[0:3])
                            # Set ROI size
                            position_Ijk=[ac.bboxLbsOrg[1], ac.bboxLbsOrg[2], ac.bboxLbsOrg[0], 1]
                            bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                            position_Ijk=[ac.bboxUbsOrg[1], ac.bboxUbsOrg[2], ac.bboxUbsOrg[0], 1]
                            bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                            self.roiNode[-1].SetSizeWorld([bboxU[0]-bboxL[0], bboxU[1]-bboxL[1], bboxU[2]-bboxL[2]])
                            #print('ROI123', len(self.roiNode))
                        else:
                            print('bboxUbsOrg123', ac.bboxUbsOrg)
                            print('bboxLbsOrg123', ac.bboxLbsOrg)
                            #dx = int((ac.bboxUbsOrg[2]+ac.bboxLbsOrg[2])/2)
                            #dy = int((ac.bboxUbsOrg[1]+ac.bboxLbsOrg[1])/2)
                            #dz = int((ac.bboxUbsOrg[0]+ac.bboxLbsOrg[0])/2)
                            dx = int((ac.bboxUbsOrg[1]+ac.bboxLbsOrg[1])/2)
                            dy = int((ac.bboxUbsOrg[2]+ac.bboxLbsOrg[2])/2)
                            dz = int((ac.bboxUbsOrg[0]+ac.bboxLbsOrg[0])/2)
                            ijkToRas = vtk.vtkMatrix4x4()
                            self.volumeNode.GetIJKToRASMatrix(ijkToRas)
                            position_Ijk=[dx, dy, dz, 1]
                            print('position_Ijk123', position_Ijk)
                            position_Ras=ijkToRas.MultiplyPoint(position_Ijk)
                            print('position_Ras123', position_Ras)
                            self.roiNode[-1].SetCenter(position_Ras[0:3])
                            # Set ROI size
                            # position_Ijk=[ac.bboxLbsOrg[1], ac.bboxLbsOrg[2], action.bboxLbsOrg[0], 1]
                            # bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                            # position_Ijk=[ac.bboxUbsOrg[1], ac.bboxUbsOrg[2], ac.bboxUbsOrg[0], 1]
                            # bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                            position_Ijk=[ac.bboxLbsOrg[1], ac.bboxLbsOrg[2], ac.bboxLbsOrg[0], 1]
                            bboxL=ijkToRas.MultiplyPoint(position_Ijk)
                            position_Ijk=[ac.bboxUbsOrg[1], ac.bboxUbsOrg[2], ac.bboxUbsOrg[0], 1]
                            bboxU=ijkToRas.MultiplyPoint(position_Ijk)
                            print('bboxL123', bboxL)
                            print('bboxU123', bboxU)
                            self.roiNode[-1].SetSizeWorld([bboxU[0]-bboxL[0], bboxU[1]-bboxL[1], bboxU[2]-bboxL[2]])
                
        # Update segment editor
        self.ui.segmentEditorWidget.setActiveEffectByName("Paint")
        

        print('Endtime', time.time() - start)
        
    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                                self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                    self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


#
# XALabelerLogic
#

class XALabelerLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# XALabelerTest
#

class XALabelerTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_XALabeler1()

    def test_XALabeler1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('XALabeler1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = XALabelerLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
