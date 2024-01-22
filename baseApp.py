from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivy.properties import StringProperty
from kivymd.uix.list import OneLineIconListItem
from kivy.lang import Builder

class IconListItem(OneLineIconListItem):
    icon = StringProperty()
    
class Task:
    def __init__(self, taskString):
        self.task = taskString

    
        

class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Red"
        taskItems = [{"viewclass": "IconListItem", "text": "Brushing teeth", "on_release": lambda x="Brushing teeth": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Cutting nails", "on_release": lambda x="Cutting nails": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Doing laundry", "on_release": lambda x="Doing laundry": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Folding clothes", "on_release": lambda x="Folding clothes": self.setItem(x)}, {"viewclass": "IconListItem", "text": "Washing dishes", "on_release": lambda x="Washing dishes": self.setItem(x)}]
        self.menu = MDDropdownMenu(
            caller=self.root.ids.taskDropDown,
            items=taskItems,
            position="center",
            width_mult=4,
        )
        self.menu.bind()

    def toTaskScreen(self, sm):
        #Add button opens card where can input task details
        #Task appears on main screen somehow
        self.root.current = "addTaskScreen"

    def toHomeScreen(self, sm):
        #Add button opens card where can input task details
        #Task appears on main screen somehow
        self.root.current = "homeScreen"

    def setItem(self, textItem):
        self.root.ids.taskDropDown.set_item(textItem)
        self.menu.dismiss()

    def setTask(self, taskString):
        task = Task(taskString)
        self.toHomeScreen()
        

if __name__ == '__main__':
    app = MainApp()
    app.run()