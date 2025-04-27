import os
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.properties import BooleanProperty, ObjectProperty 
from kivy.lang import Builder


Builder.load_file('displaybox.kv')


class DisplayBox (BoxLayout):

    main_image = ObjectProperty(None)
    top_image = ObjectProperty(None)
    bottom_image = ObjectProperty(None)
    left_image = ObjectProperty(None)
    rightimage = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MainImage(Image):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.texture = Texture.create()

    def clear_image(self):
        self.texture = Texture.create()

    def reload(self, *args):
        super().reload()
    


