import time
import os
import glob
import numpy as np

from keras.saving import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input

from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivy.properties import StringProperty
from kivymd.uix.list import OneLineIconListItem
from kivymd.uix.card import  MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.anchorlayout import AnchorLayout
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDIconButton
from kivy.uix.image import Image

#Needed for MDDropdownMenu
class IconListItem(OneLineIconListItem):
    icon = StringProperty()

class MainApp(MDApp):
    sameTaskDialog = None
    doneTaskDialog = None
    failedTaskDialog = None
    classes = ["brushing_teeth", "cutting_nails", "doing_laundry", "folding_clothes", "washing_dishes"]
    currentTaskString = None
    currentTaskCard = None

    def build(self):
        #Set colours for app
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Red"
        
        #Set tasks items for drop down menu
        taskItems = [{"viewclass": "IconListItem", "text": "Brushing teeth", "on_release": lambda x="Brushing teeth": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Cutting nails", "on_release": lambda x="Cutting nails": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Doing laundry", "on_release": lambda x="Doing laundry": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Folding clothes", "on_release": lambda x="Folding clothes": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Washing dishes", "on_release": lambda x="Washing dishes": self.setItem(x)}]
        
        #Make drop down menu
        self.menu = MDDropdownMenu(
            caller=self.root.ids.taskDropDown,
            items=taskItems,
            position="center"
        )
        self.menu.bind()



    def toTaskScreen(self):
        self.root.current = "addTaskScreen"

    def toHomeScreen(self):
        #Delete all images taken so far
        files = glob.glob('.\\takenImages\\*')
        for f in files:
            os.remove(f)
        self.currentTaskCard = None
        self.currentTaskString = None
        self.root.current = "homeScreen"

    def toCameraScreen(self):
        self.root.current = "cameraScreen"

    def toImageScreen(self, fileName):
        for child in self.root.ids.imageLayout.children:
            if (type(child) == Image):
                self.root.ids.imageLayout.remove_widget(child)
        #Add image to screen after capturing from camera
        image = Image(source=".\\takenImages\\{}".format(fileName))
        self.root.ids.imageLayout.add_widget(image)
        self.root.current = "imageScreen"



    #Sets the current item in the task drop down menu
    def setItem(self, textItem):
        self.root.ids.taskDropDown.set_item(textItem)
        self.menu.dismiss()

    def setTask(self):
        #String of current item in drop down menu
        taskString = self.root.ids.taskDropDown.current_item
        #Check if no tasks anchor layout is still on the screen and remove if it is
        if (type(self.root.ids.homeScreenWidgetLayout.children[0]) == AnchorLayout):
            self.root.ids.homeScreenWidgetLayout.remove_widget(self.root.ids.homeScreenWidgetLayout.children[0])
        
        #Make id for task card based on task string e.g. card_Brushing_teeth
        splitTask = taskString.split()
        cardId = "card_" + splitTask[0] + "_" + splitTask[1]
        cameraButtonId = "button_" + splitTask[0] + "_" + splitTask[1]
        
        #Loop through children in home screen layout and check if task already exists
        for child in self.root.ids.homeScreenWidgetLayout.children:
            #If task exists then open a dialog box that says it already exists
            if (child.id == cardId):
                self.openSameTaskDialog()
                return

        #Add camera button in the left center of the card
        cameraAnchor = AnchorLayout(anchor_x="left", anchor_y="center")
        cameraButton = MDIconButton(id=cameraButtonId, icon="camera", on_release=self.setCurrentTask)
        cameraAnchor.add_widget(cameraButton)

        #Add a delete button in the right center of the card
        deleteAnchor = AnchorLayout(anchor_x="right", anchor_y="center")
        deleteButton = MDIconButton(icon="delete", on_release=self.deleteTask)
        deleteAnchor.add_widget(deleteButton)

        #Add task label to the left top of the card
        labelAnchor = AnchorLayout(anchor_x="left", anchor_y="top")
        cardLabel = MDLabel(text=taskString, height=50)
        labelAnchor.add_widget(cardLabel)

        #Add card with above widgets ^^^^
        self.root.ids.homeScreenWidgetLayout.add_widget(MDCard(labelAnchor, cameraAnchor, deleteAnchor, id=cardId, padding=20))
        self.toHomeScreen()
        
    def setCurrentTask(self, instance):
        buttonId = instance.id
        splitId = buttonId.split("_")
        self.currentTaskString = splitId[1] + "_" + splitId[2]
        self.currentTaskCard = instance.parent.parent
        self.toCameraScreen()

    def deleteTask(self, instance):
        #Delete card (instance is the delete button on card to delete, parent is layout parent.parent is card)
        self.root.ids.homeScreenWidgetLayout.remove_widget(instance.parent.parent)
        #Check if any widgets exist in homeScreenWidgetLayout (check if no tasks)
        if (len(self.root.ids.homeScreenWidgetLayout.children) == 0):
            #Add no tasks message to home screen
            anchor =  AnchorLayout(anchor_x="center", anchor_y="center")
            label = MDLabel(id="noTasksYet", text="You have no tasks yet. Click the plus icon in the top right to add a task :)", adaptive_size=True, bold=True)
            anchor.add_widget(label)
            self.root.ids.homeScreenWidgetLayout.add_widget(anchor)



    def openSameTaskDialog(self):
        #Check if dialog exists
        if not self.sameTaskDialog:
            #Make dialog with warning and ok button that closes dialog
            self.sameTaskDialog = MDDialog(text="That task is already set. Please pick another one or return to the home screen.", buttons=[MDFlatButton(text="OK", on_release=self.closeSameTaskDialog)])
        #Open dialog
        self.sameTaskDialog.open()

    def closeSameTaskDialog(self, ah):
        self.sameTaskDialog.dismiss(force=True)
        
    def openDoneTaskDialog(self):
        #Check if dialog exists
        if not self.doneTaskDialog:
            #Make dialog with warning and ok button that closes dialog
            self.doneTaskDialog = MDDialog(text="Task completed :).", buttons=[MDFlatButton(text="OK", on_release=self.closeDoneTaskDialog)])
        #Open dialog
        self.doneTaskDialog.open()

    def closeDoneTaskDialog(self, ah):
        self.doneTaskDialog.dismiss(force=True)
        self.toHomeScreen()

    def openFailedTaskDialog(self):
        #Check if dialog exists
        if not self.failedTaskDialog:
            #Make dialog with warning and ok button that closes dialog
            self.failedTaskDialog = MDDialog(text="Task not completed, please take a picture of you doing the task or go back to home screen.", buttons=[MDFlatButton(text="OK", on_release=self.closeFailedTaskDialog)])
        #Open dialog
        self.failedTaskDialog.open()

    def closeFailedTaskDialog(self, ah):
        self.failedTaskDialog.dismiss(force=True)
        self.toCameraScreen()




    def capture(self):
        #Get camera object
        camera = self.root.ids.camera
        #Get current time
        timeStr = time.strftime("%Y%m%d_%H%M%S")
        #Set file name with current time
        fileName = "IMG_{}.png".format(timeStr)
        #Save image to takenImages folder
        camera.export_to_png(".\\takenImages\\{}".format(fileName))
        #Go to image screen
        self.toImageScreen(fileName)

    def acceptImage(self):
        model = load_model("./model.h5")
        files = glob.glob('.\\takenImages\\*')
        kerasImage = load_img(files[0])
        x = img_to_array(kerasImage)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)[0]
        maxPosition=np.argmax(pred) 
        predClass = self.classes[maxPosition]
        if (predClass == self.currentTaskString.lower()):
            self.root.ids.homeScreenWidgetLayout.remove_widget(self.currentTaskCard)
            self.openDoneTaskDialog()
        else:
            self.openFailedTaskDialog

    def deleteImage(self):
        #Delete image from folder 
        files = glob.glob('.\\takenImages\\*')
        for f in files:
            os.remove(f)
        #Take back to camera screen
        self.toCameraScreen(self.root.ids.deleteImageButton)

if __name__ == '__main__':
    app = MainApp()
    app.run()