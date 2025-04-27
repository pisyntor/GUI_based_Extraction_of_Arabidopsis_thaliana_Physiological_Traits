import os
from datetime import datetime

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty
# from kivy.uix.behaviors.compoundselection import CompoundSelectionBehavior
from kivy.lang import Builder

from tkinter import Tk, filedialog

Builder.load_file("inputbox.kv")


class InputBox(FloatLayout):

    list_box = ObjectProperty(None)
    setting_box = ObjectProperty(None)
    display_box = ObjectProperty(None)

    txt_root_input = ObjectProperty(None)
    btn_root_browse = ObjectProperty(None)
    txt_output_path = ObjectProperty(None)
    btn_output_browse = ObjectProperty(None)
    tgl_main = ObjectProperty(None)
    tgl_colour_analysis = ObjectProperty(None)
    tgl_healthy = ObjectProperty(None)
    input_root_dir = StringProperty('')
    output_dir = StringProperty('')
    chk_dataset_1 = ObjectProperty(None)
    chk_dataset_2 = ObjectProperty(None)


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def button_press_callback(self, widget):
        pass
    
    def button_release_callback(self, widget):
        
        # Show input directory selection
        if widget == self.btn_root_browse:
            self.input_root_dir = os.path.normpath(self.show_load_dialog())
            if self.input_root_dir != '':
                subdirs = sorted(os.listdir(self.input_root_dir))
                self.list_box.input_root_dir = self.input_root_dir
                self.populate_eco_list(subdirs)
        
        # Show output directory selection
        elif widget == self.btn_output_browse:
            if self.chk_dataset_1.active:
                dataset_version = 1
            else:
                dataset_version = 2
            selected_output_dir = os.path.normpath(self.show_load_dialog())
            self.output_dir = os.path.join(selected_output_dir, f'dataset_{dataset_version}') #  f'{selected_output_dir}/dataset_{dataset_version}'
            if selected_output_dir == '':
                self.output_dir = os.path.join('output', f'dataset_{dataset_version}')  # f'output/dataset_{dataset_version}'

        # Main mode pressed
        elif widget == self.tgl_main and self.tgl_main.state=='down':
            # Show the main image
            self.display_box.main_image.opacity = 1
            # Hide top and bottom image
            self.display_box.top_image.opacity = 0
            self.display_box.bottom_image.opacity = 0
            self.display_box.top_image.clear_image()
            self.display_box.bottom_image.clear_image()
            # Hide left and right image
            self.display_box.left_image.opacity = 0
            self.display_box.right_image.opacity = 0
            self.display_box.left_image.clear_image()
            self.display_box.right_image.clear_image()
            # Reload image
            self.list_box.img_selection_list.reload_image()
        
        # Colour analysis mode pressed
        elif widget == self.tgl_colour_analysis and self.tgl_colour_analysis.state=='down':
            # Hide the main image
            self.display_box.main_image.opacity = 0
            # Hide top and bottom image
            self.display_box.top_image.opacity = 0
            self.display_box.bottom_image.opacity = 0
            self.display_box.top_image.clear_image()
            self.display_box.bottom_image.clear_image()
            # Show left and right image
            self.display_box.left_image.opacity = 1
            self.display_box.right_image.opacity = 1
            self.display_box.left_image.clear_image()
            self.display_box.right_image.clear_image()
            # Reload image
            self.list_box.img_selection_list.reload_image(self.tgl_colour_analysis)

        # Graph mode mode pressed
        elif widget == self.tgl_healthy and self.tgl_healthy.state=='down':
            # Hide the main image
            self.display_box.main_image.opacity = 0
            # Show top and bottom image
            self.display_box.top_image.opacity = 1
            self.display_box.bottom_image.opacity = 1
            self.display_box.top_image.clear_image()
            self.display_box.bottom_image.clear_image()
            # Hide left and right image
            self.display_box.left_image.opacity = 0
            self.display_box.right_image.opacity = 0
            self.display_box.left_image.clear_image()
            self.display_box.right_image.clear_image()
            # Reload image
            self.list_box.eco_selection_list.reload_graph()
    
    def show_load_dialog(self):
        root = Tk()
        root.withdraw()
        dirname = os.path.normpath(filedialog.askdirectory())
        root.destroy()
        if dirname:
            return dirname
        else:
            return ''
        
    def populate_eco_list (self, ecos=[]):
        # Populate the eco list
        self.list_box.populate_eco_list(ecos)

    def enable_controls(self):
        self.txt_root_input.disabled = False
        self.btn_root_browse.disabled = False
        self.txt_output_path.disabled = False
        self.btn_output_browse.disabled = False
        self.chk_dataset_1.disabled = False
        self.chk_dataset_2.disabled = False
        self.tgl_main.disabled = False
        self.tgl_colour_analysis.disabled = False
        self.tgl_healthy.disabled = False

    def disable_controls(self):
        self.txt_root_input.disabled = True
        self.btn_root_browse.disabled = True
        self.txt_output_path.disabled = True
        self.btn_output_browse.disabled = True
        self.chk_dataset_1.disabled = True
        self.chk_dataset_2.disabled = True
        self.tgl_main.disabled = True
        self.tgl_colour_analysis.disabled = True
        self.tgl_healthy.disabled = True

    def checkbox_callback(self, widget, *args):      
        if widget == self.chk_dataset_1:
            self.setting_box.sow_date = datetime.strptime(self.setting_box.sow_dates['DS_1'],
                                                          '%Y-%m-%d').strftime('%d-%m-%Y')
            self.setting_box.screening_date = datetime.strptime(self.setting_box.screening_starts['DS_1'],
                                                                '%Y-%m-%d').strftime('%d-%m-%Y')
            self.setting_box.experiment_end_date = datetime.strptime(self.setting_box.end_dates['DS_1'],
                                                                     '%Y-%m-%d').strftime('%d-%m-%Y')
            self.reset()
        elif widget == self.chk_dataset_2:
            self.setting_box.sow_date = datetime.strptime(self.setting_box.sow_dates['DS_2'],
                                                          '%Y-%m-%d').strftime('%d-%m-%Y')
            self.setting_box.screening_date = datetime.strptime(self.setting_box.screening_starts['DS_2'],
                                                                '%Y-%m-%d').strftime('%d-%m-%Y')
            self.setting_box.experiment_end_date = datetime.strptime(self.setting_box.end_dates['DS_2'],
                                                                     '%Y-%m-%d').strftime('%d-%m-%Y')
            self.reset()

    def reset(self):
        self.input_root_dir = ''
        self.output_dir = ''
        self.list_box.eco_selection_list.clear_widgets()
        self.list_box.eco_selection_list.clear_selection()
        self.list_box.display_box.main_image.clear_image()
        self.list_box.display_box.top_image.clear_image()
        self.list_box.display_box.bottom_image.clear_image()
        self.list_box.display_box.left_image.clear_image()
        self.list_box.display_box.right_image.clear_image()
        self.setting_box.chk_selected_eco.active = True
        self.setting_box.chk_segmented.active = True


class CustomToggle(ToggleButton):
    def on_touch_down(self, *args):
        if self.state == 'down':
            return
        super(CustomToggle, self).on_touch_down(*args)


class CustomCheckBox(CheckBox):
    def on_touch_down(self, *args):
        if self.active:
            return
        super(CustomCheckBox, self).on_touch_down(*args)
