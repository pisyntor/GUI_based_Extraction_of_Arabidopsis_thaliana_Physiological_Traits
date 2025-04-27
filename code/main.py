from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
from kivy.core.window import Window
# from kivy.app import App
from kivymd.app import MDApp
from kivy.properties import ObjectProperty

from manager import Manager


class PlantApp(MDApp):
    
    title = 'Plant Stress Analyzer'
    manager = ObjectProperty(None)
    
    def build(self):
        Window.minimum_width, Window.minimum_height = (1200, 600)
        self.manager = Manager()
        return self.manager

PlantApp().run()
