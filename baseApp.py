import time
import os
import glob

from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivy.properties import StringProperty
from kivymd.uix.list import OneLineIconListItem
from kivymd.uix.card import  MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.anchorlayout import AnchorLayout
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDIconButton

class IconListItem(OneLineIconListItem):
    icon = StringProperty()

class MainApp(MDApp):
    dialog = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Red"
        taskItems = [{"viewclass": "IconListItem", "text": "Brushing teeth", "on_release": lambda x="Brushing teeth": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Cutting nails", "on_release": lambda x="Cutting nails": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Doing laundry", "on_release": lambda x="Doing laundry": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Folding clothes", "on_release": lambda x="Folding clothes": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Washing dishes", "on_release": lambda x="Washing dishes": self.setItem(x)}]
        self.menu = MDDropdownMenu(
            caller=self.root.ids.taskDropDown,
            items=taskItems,
            position="center"
        )
        self.menu.bind()

    def toTaskScreen(self):
        self.root.current = "addTaskScreen"

    def toHomeScreen(self):
        self.root.current = "homeScreen"
        files = glob.glob('.\\takenImages\\*')
        for f in files:
            os.remove(f)

    def setItem(self, textItem):
        self.root.ids.taskDropDown.set_item(textItem)
        self.menu.dismiss()

    def setTask(self):
        taskString = self.root.ids.taskDropDown.current_item
        print("Setting task")
        print(type(self.root.ids.homeScreenWidgetLayout.children[0]))
        if (type(self.root.ids.homeScreenWidgetLayout.children[0]) == AnchorLayout):
            self.root.ids.homeScreenWidgetLayout.remove_widget(self.root.ids.homeScreenWidgetLayout.children[0])
        splitTask = taskString.split()
        cardId = "card_" + splitTask[0] + "_" + splitTask[1]
        for child in self.root.ids.homeScreenWidgetLayout.children:
            if (type(child) == AnchorLayout):
                continue
            if (child.id == cardId):
                self.openDialog()
                return

        cameraAnchor = AnchorLayout(anchor_x="left", anchor_y="center")
        cameraButton = MDIconButton(icon="camera", on_release=self.takeImage)
        cameraAnchor.add_widget(cameraButton)

        deleteAnchor = AnchorLayout(anchor_x="right", anchor_y="center")
        deleteButton = MDIconButton(icon="delete", on_release=self.deleteTask)
        deleteAnchor.add_widget(deleteButton)

        labelAnchor = AnchorLayout(anchor_x="left", anchor_y="top")
        cardLabel = MDLabel(text=taskString, height=50)
        labelAnchor.add_widget(cardLabel)

        self.root.ids.homeScreenWidgetLayout.add_widget(MDCard(labelAnchor, cameraAnchor, deleteAnchor, id=cardId, padding=20))
        self.toHomeScreen()

    def openDialog(self):
        if not self.dialog:
            self.dialog = MDDialog(text="That task is already set. Please pick another one or return to the home screen.", buttons=[MDFlatButton(text="OK", on_release=self.closeDialog)])
        self.dialog.open()

    def closeDialog(self, ah):
        self.dialog.dismiss(force=True)

    def deleteTask(self, instance):
        self.root.ids.homeScreenWidgetLayout.remove_widget(instance.parent.parent)
        if (len(self.root.ids.homeScreenWidgetLayout.children) == 0):
            anchor =  AnchorLayout(anchor_x="center", anchor_y="center")
            label = MDLabel(id="noTasksYet", text="You have no tasks yet. Click the plus icon in the top right to add a task :)", adaptive_size=True, bold=True)
            anchor.add_widget(label)
            self.root.ids.homeScreenWidgetLayout.add_widget(anchor)
            
    def takeImage(self, ah):
        self.root.current = "cameraScreen"
    
    def capture(self):
        camera = self.root.ids.camera
        timeStr = time.strftime("%Y%m%d_%H%M%S")
        fileName = "IMG_{}.png".format(timeStr)
        camera.export_to_png(".\\takenImages\\{}".format(fileName))
        print("Captured")

if __name__ == '__main__':
    app = MainApp()
    app.run()