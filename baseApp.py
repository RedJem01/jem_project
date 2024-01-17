from kivymd.app import MDApp

class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Red"
    
    def callback(self, sm):
        #Add button opens card where can input task details
        #Task appears on main screen somehow
        print(self.root.current)
        print("changing screens")
        self.root.current = "addTaskScreen"
        print(self.root.current)
    

if __name__ == '__main__':
    app = MainApp()
    app.run()