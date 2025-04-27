from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout


Builder.load_file('manager.kv')

class Manager(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                
