import os
from pathlib import Path
from functools import partial
from threading import Thread, Condition
# from PIL import Image

from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.lang import Builder
from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.behaviors.compoundselection import CompoundSelectionBehavior
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label

from analyzer import process_piecharts

Builder.load_file("listbox.kv")


class ListBox(FloatLayout):

    input_box = ObjectProperty(None)
    setting_box = ObjectProperty(None)
    display_box = ObjectProperty(None)

    txt_root_input = ObjectProperty(None)
    btn_root_browse = ObjectProperty(None)
    txt_output_path = ObjectProperty(None)
    btn_output_browse = ObjectProperty(None)
    eco_selection_list = ObjectProperty(None)
    rep_selection_list = ObjectProperty(None)
    img_selection_list = ObjectProperty(None)
    eco_selection_scroll = ObjectProperty(None)
    img_scroll = ObjectProperty(None)

    input_root_dir = StringProperty('')
    output_dir = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass
        
    def populate_eco_list (self, ecos=[]):
        # Clearing the lists
        self.eco_selection_list.clear_widgets()
        self.eco_selection_list.clear_selection()
        # Giving the root directory to eco selection list
        self.eco_selection_list.root_dir = os.path.normpath(self.input_root_dir)
        for eco in ecos:
            self.eco_selection_list.add_widget(MyListItem(text=eco))
        self.eco_selection_scroll.scroll_y = 1


class EcoListLayout (FocusBehavior, CompoundSelectionBehavior, GridLayout):

    # A reference to setting box to connect to setting box and image type
    list_box = ObjectProperty(None)
    scroll = ObjectProperty(None)
    selected_names = []
    last_selected_node = None
    root_dir = ''
    rep_selection_list = ObjectProperty(None)
    rep_selection_scroll = ObjectProperty(None)
    is_node_selected = BooleanProperty(False)
    condition = Condition()

    def keyboard_on_key_down(self, keyboard, keycode, text, modifiers):
        if not self.list_box.setting_box.is_processing:
            if super(EcoListLayout, self).keyboard_on_key_down(keyboard, keycode, text, modifiers):
                return True
            if self.select_with_key_down(keyboard, keycode, text, modifiers):
                return True
            return False

    def keyboard_on_key_up(self, keyboard, keycode):
        if not self.list_box.setting_box.is_processing:
            if super(EcoListLayout, self).keyboard_on_key_up(keyboard, keycode):
                return True
            if self.select_with_key_up(keyboard, keycode):
                return True
            return False

    def add_widget(self, widget):
        super().add_widget(widget)
        widget.bind(on_touch_down = self.widget_touch_down)
    
    def widget_touch_down(self, widget, touch):
        if not self.list_box.setting_box.is_processing:
            if widget.collide_point(*touch.pos):
                self.select_with_touch(widget, touch)
    
    def widget_touch_up(self, widget, touch):
        if not self.list_box.setting_box.is_processing:
            if self.collide_point(*touch.pos) and (not (widget.collide_point(*touch.pos) or self.touch_multiselect)):
                self.deselect_node(widget)
    
    def select_node(self, node, *args):
        super().select_node(node)
        with self.condition:
            node.select()
            self.last_selected_node = node
            if self.height > self.scroll.height:
                self.scroll.scroll_to(node)
            self.selected_names.append(node.text)
            # Populate rep selection list
            self.populate_rep_list()
            self.is_node_selected = True
            self.condition.notify_all()
        # Load graph
        if self.list_box.input_box.tgl_healthy.state == 'down':
            Clock.schedule_once(partial(self.display_graph, node.text), -1)
        return 
    
    def deselect_node(self, node):
        super().deselect_node(node)
        node.deselect()
        # Check if nothing is selected
        if len(self.selected_nodes) == 0:
            self.is_node_selected = False
    
    def clear_selection(self, widget=None):
        self.rep_selection_list.clear_widgets()
        self.rep_selection_list.clear_selection()
        return super().clear_selection()

    def on_selected_nodes(self, grid, nodes):
        pass

    def populate_rep_list (self):
        rep_dirs = sorted(os.listdir(os.path.join(os.path.normpath(self.root_dir), self.last_selected_node.text)))
        if rep_dirs != []:
            self.rep_selection_list.clear_widgets()
            self.rep_selection_list.clear_selection()
            # Giving the root directory to rep selection list
            self.rep_selection_list.root_dir = os.path.join(os.path.normpath(self.root_dir),
                                                            self.last_selected_node.text)
            for dir in rep_dirs:
                self.rep_selection_list.add_widget(MyListItem(text=dir))
        self.rep_selection_scroll.scroll_y = 1

    def display_graph(self, class_name, *args):
        # Display the graph
        output_dir = os.path.normpath(self.list_box.input_box.output_dir)
        if output_dir == '':
            output_dir = 'output'
        path = os.path.join(output_dir, class_name)
        stress_graph_path = os.path.join(path, "plots", "Stress_by_date.png")
        healthy_graph_path = os.path.join(path, "plots", "Healthy_by_date.png")
        # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
        if os.path.isfile('\\\\?\\' + os.path.abspath(stress_graph_path)):
            self.list_box.display_box.bottom_image.source = '\\\\?\\' + os.path.abspath(stress_graph_path)
            self.list_box.display_box.bottom_image.reload()
        else:
            print ('Stress graph data not found')
            self.list_box.display_box.bottom_image.clear_image()
        # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
        if os.path.isfile('\\\\?\\' + os.path.abspath(healthy_graph_path)):
            self.list_box.display_box.top_image.source = '\\\\?\\' + os.path.abspath(healthy_graph_path)
            self.list_box.display_box.top_image.reload()
        else:
            print ('Healthy graph data not found')
            self.list_box.display_box.top_image.clear_image()

    def reload_graph(self, *args):
        # Reload the graph
        if self.last_selected_node is not None:
            output_dir = os.path.normpath(self.list_box.input_box.output_dir)
            if output_dir == '':
                output_dir = 'output'
            path = os.path.join(output_dir, self.last_selected_node.text)
            stress_graph_path = os.path.join(path, "plots", "Stress_by_date.png")
            healthy_graph_path = os.path.join(path, "plots", "Healthy_by_date.png")
            # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
            if os.path.isfile('\\\\?\\' + os.path.abspath(stress_graph_path)):
                self.list_box.display_box.bottom_image.source = '\\\\?\\' + os.path.abspath(stress_graph_path)
                self.list_box.display_box.bottom_image.reload()
            else:
                print ('Stress graph data not found')
                self.list_box.display_box.bottom_image.clear_image()
            # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
            if os.path.isfile('\\\\?\\' + os.path.abspath(healthy_graph_path)):
                self.list_box.display_box.top_image.source = '\\\\?\\' + os.path.abspath(healthy_graph_path)
                self.list_box.display_box.top_image.reload()
            else:
                print ('Healthy graph data not found')
                self.list_box.display_box.top_image.clear_image()


class RepListLayout (FocusBehavior, CompoundSelectionBehavior, GridLayout):

    # A reference to setting box to connect to setting box and image type
    list_box = ObjectProperty(None)
    scroll = ObjectProperty(None)
    last_selected_node = None
    root_dir = ''
    is_node_selected = BooleanProperty(False)
    img_selection_list = ObjectProperty(None)
    img_scroll = ObjectProperty(None)

    condition = Condition()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if not self.list_box.setting_box.is_processing:
            if super().keyboard_on_key_down(window, keycode, text, modifiers):
                return True
            if self.select_with_key_down(window, keycode, text, modifiers):
                return True
            return False

    def keyboard_on_key_up(self, window, keycode):
        if not self.list_box.setting_box.is_processing:
            if super().keyboard_on_key_up(window, keycode):
                return True
            if self.select_with_key_up(window, keycode):
                return True
            return False

    def add_widget(self, widget):
        super().add_widget(widget)
        widget.bind(on_touch_down = self.widget_touch_down)
    
    def widget_touch_down(self, widget, touch):
        if not self.list_box.setting_box.is_processing:
            if widget.collide_point(*touch.pos):
                self.select_with_touch(widget, touch)
    
    def select_node(self, node, *args):
        super().select_node(node)
        with self.condition:
            node.select()
            self.last_selected_node = node
            if self.height > self.scroll.height:
                self.scroll.scroll_to(node)
            # Populate img selection list
            self.populate_img_list()
            self.is_node_selected = True
            self.condition.notify_all()
            Clock.schedule_once(self.list_box.img_selection_list.clear_selection, -1)
            Clock.schedule_once(partial(self.list_box.img_selection_list.select_node,
                                        self.list_box.img_selection_list.children[-1]), -1)
        return 
        
    def deselect_node(self, node):
        super().deselect_node(node)
        node.deselect()
        # Check if nothing is selected
        if len(self.selected_nodes) == 0:
            self.is_node_selected = False
    
    def clear_selection(self, widget=None):
        self.img_selection_list.clear_widgets()
        self.img_selection_list.clear_selection()
        return super().clear_selection()

    def on_selected_nodes(self,grid,nodes):
        self.list_box.setting_box.add_rep_selection(nodes, self.list_box.setting_box.chk_selected_reps)
        
    def populate_img_list (self):

        try:
            img_dirs = sorted(os.listdir(os.path.join(os.path.normpath(self.root_dir),
                                                      self.last_selected_node.text,
                                                      'segmented_images')))
            if img_dirs != []:
                self.img_selection_list.clear_widgets()
                self.img_selection_list.clear_selection()
                # Giving the root directory to rep selection list
                self.img_selection_list.root_dir = os.path.join(os.path.normpath(self.root_dir),
                                                                self.last_selected_node.text)
                for idx, dir in enumerate(img_dirs):
                    self.img_selection_list.add_widget(MyListItem(text=str(idx+1), file_name=dir))
            self.img_scroll.scroll_y = 1

        except Exception as e:
            print (e)


class ImgListLayout (FocusBehavior, CompoundSelectionBehavior, GridLayout):

    # A reference to input box to connect to dislay box
    list_box = ObjectProperty(None)
    scroll = ObjectProperty(None)
    eco_selection_list = ObjectProperty(None)
    rep_selection_list = ObjectProperty(None)
    last_selected_node = None
    root_dir = ''
    is_node_selected = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if not self.list_box.setting_box.is_processing:
            super().keyboard_on_key_down(window, keycode, text, modifiers)
                #return True
            if self.select_with_key_down(window, keycode, text, modifiers):
                return True
            return False

    def keyboard_on_key_up(self, window, keycode):
        if not self.list_box.setting_box.is_processing:
            super().keyboard_on_key_up(window, keycode)
                #return True
            if self.select_with_key_up(window, keycode):
                return True
            return False

    def add_widget(self, widget):
        super().add_widget(widget)
        widget.bind(on_touch_down = self.widget_touch_down)
    
    def widget_touch_down(self, widget, touch):
        if not self.list_box.setting_box.is_processing:
            if widget.collide_point(*touch.pos):
                self.select_with_touch(widget, touch)
    

    def select_node(self, node, *args, **kwargs):
    
        super().select_node(node)
        node.select()
        self.last_selected_node = node
        if self.height > self.scroll.height:
            self.scroll.scroll_to(node)
        img_path = os.path.join(os.path.normpath(self.root_dir), 'segmented_images', node.file_name)
        
        # Load image
        if self.list_box.setting_box.chk_segmented.active:
            
            # Segmentation 
            ## Check the mode selection
            if not self.list_box.input_box.tgl_colour_analysis.state == 'down':
                ## Display the segmentation only
                Clock.schedule_once(partial(self.load_image, img_path), -1)
                
            else:
                ## Process the pie
                output_dir = os.path.normpath(self.list_box.input_box.output_dir)
                if output_dir == '':
                    output_dir = 'output'
                    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                    os.makedirs('\\\\?\\' + os.path.abspath(output_dir), exist_ok=True)
                ## Check if pie already exist
                output_path = img_path.replace(os.path.normpath(self.list_box.input_root_dir),
                                               os.path.abspath(output_dir))
                # path_parts = output_path.split(os.sep)
                path_parts = list(Path(output_path).parts)
                path_parts[-2] = "pie_charts"
                output_path = os.path.join(
                    os.path.join(*path_parts[:-3]),
                    path_parts[-2],
                    path_parts[-3],
                    path_parts[-1]
                )
                # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                if not os.path.exists('\\\\?\\' + os.path.abspath(output_path)):
                    # Process and display the pie chart in new thread
                    def __process_pie():
                        piechart_path = process_piecharts(img_path,
                                                          output_dir,
                                                          os.path.normpath(self.list_box.input_root_dir))
                        ## Display the pie
                        Clock.schedule_once(partial(self.load_pie, img_path, piechart_path), -1)
                        # Unlock the selection
                        self.list_box.setting_box.is_processing = False
                    # Lock the selection
                    self.list_box.setting_box.is_processing = True
                    # Process the pie
                    process_pie_t = Thread(target=__process_pie)
                    process_pie_t.start()
                else:
                    piechart_path = process_piecharts(img_path,
                                                      output_dir,
                                                      os.path.normpath(self.list_box.input_root_dir))
                    ## Display the pie
                    Clock.schedule_once(partial(self.load_pie, img_path, piechart_path), -1)
        
        elif self.list_box.setting_box.chk_mask.active:
            # Mask
            mask_file_name = (os.path.splitext(node.file_name)[0]).split('_')[:-1] 
            mask_file_name = f"{'_'.join(mask_file_name)}_mask.png"     
            mask_path = os.path.join(os.path.normpath(self.root_dir), 'masks', mask_file_name)
            ## Check the mode selection
            if not self.list_box.input_box.tgl_colour_analysis.state == 'down':
                ## Display the segmentation only
                Clock.schedule_once(partial(self.load_image, mask_path), -1)
            else:
                ## Process the pie
                output_dir = os.path.normpath(self.list_box.input_box.output_dir)
                if output_dir == '':
                    output_dir = 'output'
                    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                    os.makedirs('\\\\?\\' + os.path.abspath(output_dir), exist_ok=True)
                ## Check if pie already exist
                output_path = img_path.replace(os.path.normpath(self.list_box.input_root_dir),
                                               os.path.abspath(output_dir))
                path_parts = list(Path(output_path).parts)
                path_parts[-2] = "pie_charts"
                output_path = os.path.join(
                    os.path.join(*path_parts[:-3]),
                    path_parts[-2],
                    path_parts[-3],
                    path_parts[-1]
                )
                # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                if not os.path.exists('\\\\?\\' + os.path.abspath(output_path)):
                    # Process and display the pie chart in new thread
                    def __process_pie():
                        piechart_path = process_piecharts(img_path,
                                                          output_dir,
                                                          os.path.normpath(self.list_box.input_root_dir))
                        ## Display the pie
                        Clock.schedule_once(partial(self.load_pie, mask_path, piechart_path), -1)
                        # Unlock the selection
                        self.list_box.setting_box.is_processing = False
                    # Lock the selection
                    self.list_box.setting_box.is_processing = True
                    # Process the pie
                    process_pie_t = Thread(target=__process_pie)
                    process_pie_t.start()
                else:
                    piechart_path = process_piecharts(img_path,
                                                      output_dir,
                                                      os.path.normpath(self.list_box.input_root_dir))
                    ## Display the pie
                    Clock.schedule_once(partial(self.load_pie, mask_path, piechart_path), -1)
        
        elif self.list_box.setting_box.chk_stress.active:
            # Stress
            ## Check the mode selection
            if not self.list_box.input_box.tgl_colour_analysis.state == 'down':
                try:
                    # Check if this is triggered from the processing thread
                    img_path = kwargs['img_path']
                    Clock.schedule_once(partial(self.load_image, img_path), -1)
                except:
                    # Try to look from output folder first
                    output_dir = os.path.normpath(self.list_box.input_box.output_dir)
                    if output_dir == '':
                        output_dir = 'output'
                    stress_path = os.path.join(output_dir, 
                                            self.eco_selection_list.last_selected_node.text,
                                            'stressed_area', 
                                            self.rep_selection_list.last_selected_node.text,
                                            node.file_name)
                    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                    if os.path.isfile('\\\\?\\' + os.path.abspath(stress_path)):
                        Clock.schedule_once(partial(self.load_image, stress_path), -1)
                    else:
                        ### Output file does not exist. Load the segmented image
                        img_path = os.path.join(os.path.normpath(self.root_dir), 'segmented_images', node.file_name)
                        Clock.schedule_once(partial(self.load_image, img_path), -1)
            else:
                ## Process the pie
                output_dir = os.path.normpath(self.list_box.input_box.output_dir)
                if output_dir == '':
                    output_dir = 'output'
                    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                    os.makedirs('\\\\?\\' + os.path.abspath(output_dir), exist_ok=True)
                ## Check if pie already exist
                output_path = img_path.replace(os.path.normpath(self.list_box.input_root_dir),
                                               os.path.abspath(output_dir))
                path_parts = list(Path(output_path).parts)
                path_parts[-2] = "pie_charts"
                output_path = os.path.join(
                    os.path.join(*path_parts[:-3]),
                    path_parts[-2],
                    path_parts[-3],
                    path_parts[-1]
                )
                # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                if not os.path.exists('\\\\?\\' + os.path.abspath(output_path)):
                    # Process and display the pie chart in new thread
                    def __process_pie():
                        piechart_path = process_piecharts(img_path,
                                                          output_dir,
                                                          os.path.normpath(self.list_box.input_root_dir))
                        ## Display the pie
                        Clock.schedule_once(partial(self.load_pie, img_path, piechart_path), -1)
                        # Unlock the selection
                        self.list_box.setting_box.is_processing = False
                    # Lock the selection
                    self.list_box.setting_box.is_processing = True
                    # Process the pie
                    process_pie_t = Thread(target=__process_pie)
                    process_pie_t.start()
                else:
                    piechart_path = process_piecharts(img_path,
                                                      output_dir,
                                                      os.path.normpath(self.list_box.input_root_dir))
                    ## Display the pie
                    Clock.schedule_once(partial(self.load_pie, img_path, piechart_path), -1)

        self.is_node_selected = True
        

    def deselect_node(self, node):
        super().deselect_node(node)
        node.deselect()
        # Check if nothing is selected
        if len(self.selected_nodes) == 0:
            self.is_node_selected = False
    
    def clear_selection(self, widget=None):
        self.last_selected_node = None
        self.list_box.display_box.main_image.clear_image()
        return super().clear_selection()

    def on_selected_nodes(self, grid, nodes):
        pass

    def load_image(self, img_path, *args):
        self.list_box.display_box.main_image.source = '\\\\?\\' + os.path.abspath(img_path)
        self.list_box.display_box.main_image.reload()


    def reload_image(self, *args):

        if self.last_selected_node is not None:

            img_path = os.path.join(self.root_dir, 'segmented_images', self.last_selected_node.file_name)
            # Load image
            if self.list_box.setting_box.chk_segmented.active:
                # Segmentation 
                ## Check the mode selection
                if not self.list_box.input_box.tgl_colour_analysis.state == 'down':
                    ## Display the segmentation only
                    Clock.schedule_once(partial(self.load_image, img_path), -1)
                else:
                    ## Process the pie
                    output_dir = os.path.normpath(self.list_box.input_box.output_dir)
                    if output_dir == '':
                        output_dir = 'output'
                        # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                        os.makedirs('\\\\?\\' + os.path.abspath(output_dir), exist_ok=True)
                    piechart_path = process_piecharts(img_path,
                                                      output_dir,
                                                      os.path.normpath(self.list_box.input_root_dir),
                                                      force_create=False)
                    if piechart_path is not None:
                        ## Display the pie
                        Clock.schedule_once(partial(self.load_pie, img_path, piechart_path), -1)
            
            elif self.list_box.setting_box.chk_mask.active:
                # Mask
                mask_file_name = (os.path.splitext(self.last_selected_node.file_name)[0]).split('_')[:-1] 
                mask_file_name = f"{'_'.join(mask_file_name)}_mask.png"     
                mask_path = os.path.join(self.root_dir, 'masks', mask_file_name)
                ## Check the mode selection
                if not self.list_box.input_box.tgl_colour_analysis.state == 'down':
                    ## Display the segmentation only
                    Clock.schedule_once(partial(self.load_image, mask_path), -1)
                else:
                    ## Process the pie
                    output_dir = os.path.normpath(self.list_box.input_box.output_dir)
                    if output_dir == '':
                        output_dir = 'output'
                        # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                        os.makedirs('\\\\?\\' + os.path.abspath(output_dir), exist_ok=True)
                    piechart_path = process_piecharts(img_path,
                                                      output_dir,
                                                      os.path.normpath(self.list_box.input_root_dir),
                                                      force_create=False)
                    if piechart_path is not None:
                        ## Display the pie
                        Clock.schedule_once(partial(self.load_pie, mask_path, piechart_path), -1)

            elif self.list_box.setting_box.chk_stress.active:
                # Stress
                ## Check the mode selection
                if not self.list_box.input_box.tgl_colour_analysis.state == 'down':
                    # Try to look from output folder first
                    output_dir = os.path.normpath(self.list_box.input_box.output_dir)
                    if output_dir == '':
                        output_dir = 'output'
                    stress_path = os.path.join(output_dir, 
                                               self.eco_selection_list.last_selected_node.text,
                                               'stressed_area', 
                                               self.rep_selection_list.last_selected_node.text,
                                               self.last_selected_node.file_name)
                    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                    if os.path.isfile('\\\\?\\' + os.path.abspath(stress_path)):
                        Clock.schedule_once(partial(self.load_image, stress_path), -1)
                    else:
                        ### Output file does not exist. Load the segmented image
                        Clock.schedule_once(partial(self.load_image, img_path), -1)
                else:
                    ## Process the pie
                    output_dir = os.path.normpath(self.list_box.input_box.output_dir)
                    if output_dir == '':
                        output_dir = 'output'
                        # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                        os.makedirs('\\\\?\\' + os.path.abspath(output_dir), exist_ok=True)
                    piechart_path = process_piecharts(img_path,
                                                      output_dir,
                                                      os.path.normpath(self.list_box.input_root_dir),
                                                      force_create=False)
                    if piechart_path is not None:
                        ## Display the pie
                        Clock.schedule_once(partial(self.load_pie, img_path, piechart_path), -1)

    def load_pie(self, seg_path, pie_path, *args):
        self.list_box.display_box.left_image.source = '\\\\?\\' + os.path.abspath(seg_path)
        self.list_box.display_box.right_image.source = '\\\\?\\' + os.path.abspath(pie_path)
        self.list_box.display_box.left_image.reload()
        self.list_box.display_box.right_image.reload()


class MyListItem(Label):
    
    def __init__(self, file_name='', **kwargs):
        super().__init__(**kwargs)
        self.file_name = file_name
    
    def update_rect(self, color, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def select(self, color=(0, 0.47, 0.84), *args):
        self.color = (1, 1, 1)
        with self.canvas.before:
            Color(color[0], color[1], color[2])
            self.rect = Rectangle(pos=self.pos, size=self.size)
    
    def deselect(self, color=(1.0, 1.0, 1.0)):
        self.color = (0, 0, 0)
        with self.canvas.before:
            Color(color[0], color[1], color[2])
            self.rect = Rectangle(pos=self.pos, size=self.size)
