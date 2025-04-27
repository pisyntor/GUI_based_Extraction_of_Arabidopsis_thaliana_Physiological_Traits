import gc
from functools import partial
from threading import Thread, Condition

from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.checkbox import CheckBox
from kivy.properties import ObjectProperty, StringProperty
from kivymd.uix.picker import MDDatePicker
from kivy.clock import Clock


from analyzer import *

Builder.load_file('settingbox.kv')


class SettingBox(FloatLayout):

    input_box = ObjectProperty(None)
    list_box = ObjectProperty(None)
    btn_sow_date = ObjectProperty(None)
    btn_screening_date = ObjectProperty(None)
    btn_experiment_end_date = ObjectProperty(None)
    chk_segmented = ObjectProperty(None)
    lbl_segmented = ObjectProperty(None)
    chk_mask = ObjectProperty(None)
    lbl_mask = ObjectProperty(None)
    chk_stress = ObjectProperty(None)
    lbl_stress = ObjectProperty(None)
    sow_date = StringProperty('')
    screening_date = StringProperty('')
    experiment_end_date = StringProperty('')
    btn_reset = ObjectProperty(None)
    btn_reset_selection = ObjectProperty(None)
    btn_process = ObjectProperty(None)
    chk_selected_reps = ObjectProperty(None)
    txt_selected_reps = ObjectProperty(None)
    chk_selected_eco = ObjectProperty(None)
    chk_all_classes = ObjectProperty(None)
    lbl_selected_eco = ObjectProperty(None) 
    lbl_selected_reps = ObjectProperty(None)
    lbl_all_classes = ObjectProperty(None)
    
    selected_reps = []
    selected_reps_show = []
    # Processing flag
    is_processing = False
    processing_condition = Condition()

    # Dataset specific data
    calibration_factors = {"DS_1": 0.13715, "DS_2": 0.14690}
    sow_dates = {"DS_1": "2022-04-28", "DS_2": "2022-07-22"}
    screening_starts = {"DS_1": "2022-05-11", "DS_2": "2022-08-02"}
    end_dates =  {"DS_1": "2022-06-07", "DS_2": "2022-09-15"}
    screening_das = {"DS_1": 13, "DS_2": 11}


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize dates
        self.sow_date = datetime.strptime(self.sow_dates['DS_1'], '%Y-%m-%d').strftime('%d-%m-%Y') 
        self.screening_date = datetime.strptime(self.screening_starts['DS_1'], '%Y-%m-%d').strftime('%d-%m-%Y') 
        self.experiment_end_date = datetime.strptime(self.end_dates['DS_1'], '%Y-%m-%d').strftime('%d-%m-%Y') 
        self.process_thread = Thread()


    def checkbox_callback(self, widget, *args):      
        if ((widget == self.chk_segmented) or (widget == self.chk_mask) or (widget == self.chk_stress))\
                and widget.active == True:
            Clock.schedule_once(self.list_box.img_selection_list.reload_image, 0)
        elif widget == self.chk_selected_reps:
            self.get_rep_selection(widget)


    def button_press_callback(self, widget):
        if widget == self.lbl_segmented:
            self.chk_segmented.active = True
        elif widget == self.lbl_mask:
            self.chk_mask.active = True
        elif widget == self.lbl_stress:
            self.chk_stress.active = True
        elif widget == self.lbl_all_classes:
            self.chk_all_classes.active = True
        elif widget == self.lbl_selected_eco:
            self.chk_selected_eco.active = True
        elif widget == self.lbl_selected_reps:
            if not self.chk_selected_reps.active == True:
                self.chk_selected_reps.active = True    
            else:
                self.chk_selected_reps.active = False


    def button_release_callback(self, button):
        if button == self.btn_sow_date:
            self.show_date_time_picker(button=button)
        elif button == self.btn_screening_date:
            self.show_date_time_picker(button=button)
        elif button == self.btn_experiment_end_date:
            self.show_date_time_picker(button=button)
        elif button == self.btn_reset_selection:
            self.reset_selection()
        elif button == self.btn_reset:
            self.reset_all()
        elif button == self.btn_process:
            if not self.process_thread.is_alive():
                self.process_thread = Thread(target=self.process_all_images)
                self.process_thread.daemon  = True
                self.process_thread.start()


    def show_date_time_picker(self, button):

        def get_date_time(instance, value, *args):
            
            if button == self.btn_sow_date:
                self.sow_date = value.strftime('%d-%m-%y') 
            elif button == self.btn_screening_date:
                self.screening_date = value.strftime('%d-%m-%y') 
            elif button == self.btn_experiment_end_date:
                self.experiment_end_date = value.strftime('%d-%m-%y') 

        date_dialog = MDDatePicker()
        date_dialog.bind(on_save=get_date_time)
        date_dialog.open()


    def reset_selection(self, *args):
        self.chk_selected_reps.active = False


    def reset_all(self):
        self.list_box.eco_selection_list.clear_widgets()
        self.list_box.eco_selection_list.clear_selection()
        self.list_box.display_box.main_image.clear_image()
        self.list_box.display_box.top_image.clear_image()
        self.list_box.display_box.bottom_image.clear_image()
        self.list_box.display_box.left_image.clear_image()
        self.list_box.display_box.right_image.clear_image()
        self.input_box.input_root_dir = ''
        self.input_box.output_dir = ''
        self.chk_selected_eco.active = True
        self.chk_segmented.active = True


    def get_rep_selection(self, checkbox):

        if checkbox.active:
            pass
        else:
            self.selected_reps.clear()
            self.selected_reps_show.clear()
            self.txt_selected_reps.text = ''


    def add_rep_selection(self, nodes, checkbox):
        if checkbox.active:
            for node in nodes:
                if not node.text in self.selected_reps:
                    self.selected_reps.append(node.text)
                    self.selected_reps_show.append((node.text).split('_')[-1])
                    reps_str = ', '.join(self.selected_reps_show)
                    self.txt_selected_reps.text = reps_str


    def process_all_images(self):

        self.is_processing = True
        
        # Checking the stress radio button
        self.chk_stress.active = True
        # Disabling controls
        self.disable_controls()
        self.input_box.disable_controls()

        # Input root directory
        input_root_dir = os.path.normpath(self.input_box.input_root_dir)
        
        # Getting dataset version
        if self.input_box.chk_dataset_1.active:
            dataset_version = 1
        else:
            dataset_version = 2

        # Checking output directory. If not specified, create default output directory
        output_root_dir = os.path.normpath(self.input_box.output_dir)
        # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
        os.makedirs('\\\\?\\' + os.path.abspath(output_root_dir), exist_ok=True)

        if input_root_dir:

            if self.chk_all_classes.active:
                eco_nodes = self.list_box.eco_selection_list.children
                all_classes = sorted([eco_node.text for eco_node in eco_nodes])
            elif self.chk_selected_eco.active:
                eco_nodes = self.list_box.eco_selection_list.selected_nodes
                all_classes = sorted([eco_node.text for eco_node in eco_nodes])

            img_nodes = self.list_box.img_selection_list.children

            for class_name in all_classes:

                # Skip ds store
                if class_name == ".DS_Store":
                    continue

                class_path = os.path.join(input_root_dir, class_name)
                # print(f' Processing {class_name}')

                # Select the eco being processed
                for eco_node in eco_nodes:
                    if class_name == eco_node.text:
                        Clock.schedule_once(self.list_box.eco_selection_list.clear_selection, 0)
                        Clock.schedule_once(partial(self.list_box.eco_selection_list.select_node, eco_node), 0)
                        # Wait for rep selection list to be loaded first
                        with self.list_box.eco_selection_list.condition:
                            self.list_box.eco_selection_list.condition.wait()
                        break

                # Getting predefined healthy reps
                if dataset_version == 1:
                    if class_name in predefined_data:
                        list_reps = predefined_data[class_name]
                else:
                    if class_name in predefined_data_2:
                        list_reps = predefined_data_2[class_name]
                healthy_reps = list_reps["healthy"]

                # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                if os.path.isdir('\\\\?\\' + os.path.abspath(class_path)):

                    if not self.chk_selected_reps.active:
                        # All reps
                        nodes = self.list_box.rep_selection_list.children
                        all_reps = sorted([node.text for node in nodes])
                    else:
                        all_reps = self.selected_reps

                    all_reps = sorted(all_reps, key=extract_rep_name)

                    # Iterate the reps
                    rep_nodes = self.list_box.rep_selection_list.children
                    all_array_stress = {}
                    all_array_healthy = {}
                    for rep_name in all_reps:                        
                        
                        single_array_stress = dict()
                        single_array_healthy = dict()

                        # Skip ds store
                        if rep_name == ".DS_Store":
                            continue

                        # Select the rep being processed
                        for rep_node in rep_nodes:
                            if rep_name == rep_node.text:
                                Clock.schedule_once(self.list_box.rep_selection_list.clear_selection, 0)
                                Clock.schedule_once(partial(self.list_box.rep_selection_list.select_node, rep_node), 0)
                                break

                        rep_path = os.path.join(class_path, rep_name)
                        # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                        if os.path.isdir('\\\\?\\' + os.path.abspath(rep_path)):
                            segmented_images_path = os.path.join(
                                rep_path, "segmented_images"
                            )
                            # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                            if os.path.exists('\\\\?\\' + os.path.abspath(segmented_images_path)):
                                image_list = sorted(
                                    os.listdir('\\\\?\\' + os.path.abspath(segmented_images_path)),
                                    key=extract_timestamp_from_filename_only,
                                )
                                
                                # Image nodes
                                img_nodes = self.list_box.img_selection_list.children
                                
                                # Iterate on images
                                for i, img_name in enumerate(image_list):

                                    img_path = os.path.join(segmented_images_path, img_name)

                                    if self.input_box.tgl_main.state == 'down' \
                                            or self.input_box.tgl_healthy.state == 'down':

                                        # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                                        img = Image.open('\\\\?\\' + os.path.abspath(img_path))
                                        
                                        if rep_name in healthy_reps:
                                            processed_img = img
                                            # Convert to grayscale
                                            gray_image = cv2.cvtColor(
                                                np.array(img), cv2.COLOR_BGR2GRAY
                                            )

                                            # Determine the cal factor based on dataset version
                                            if dataset_version == 1:
                                                # Dataset version 1
                                                cal_factor = self.calibration_factors['DS_1']
                                            else:
                                                # Dataset version 2
                                                cal_factor = self.calibration_factors['DS_2']
                                            
                                            # Calculate the total number of non-zero pixels
                                            healthy = (
                                                cv2.countNonZero(gray_image)
                                                * cal_factor
                                                * cal_factor
                                            )
                                            stressed = 0.0
                                        
                                        else:    
                                            num_clusters, lower_green, upper_green = (
                                                load_slider_values(img_path)
                                            )
                                            processed_img, stressed, healthy = (
                                                update_image(
                                                    img,
                                                    num_clusters,
                                                    lower_green,
                                                    upper_green,
                                                    dataset_version
                                                )
                                            )

                                        date = str(
                                            extract_timestamp_from_filename_only(img_name)
                                        )
                                            
                                        update_dict_with_average(
                                            single_array_healthy, date, healthy
                                        )
                                        update_dict_with_average(
                                            single_array_stress, date, stressed
                                        )
                                        output_path = img_path.replace(
                                            os.path.abspath(input_root_dir),
                                            os.path.abspath(output_root_dir)
                                        )
                                        path_parts = list(Path(output_path).parts)
                                        path_parts[-2] = "stressed_area"
                                        output_path = os.path.join(
                                            os.path.join(*path_parts[:-3]),
                                            path_parts[-2],
                                            path_parts[-3],
                                            path_parts[-1]
                                        )

                                        # Writing processed image
                                        # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
                                        os.makedirs('\\\\?\\' + os.path.abspath(os.path.dirname(output_path)),
                                                    exist_ok=True)
                                        processed_img.save('\\\\?\\' + os.path.abspath(output_path))
                                        
                                        # Select the image being processed
                                        for img_node in img_nodes:
                                            if img_name == img_node.file_name:
                                                Clock.schedule_once(
                                                    partial(
                                                        self.list_box.img_selection_list.select_node,
                                                        img_node,
                                                        img_path=output_path),
                                                    0)
                                                break
                                        
                                        # Colour analysis
                                        if self.chk_all_classes.active == True:
                                            piechart_path = process_piecharts(img_path, output_root_dir, input_root_dir)
                                        
                                        del img
                                        del processed_img
                                        gc.collect()

                                    elif self.input_box.tgl_colour_analysis.state == 'down':
                                        # Colour analysis
                                        piechart_path = process_piecharts(img_path, output_root_dir, input_root_dir)
                                        ## Display the pie
                                        Clock.schedule_once(partial(self.list_box.img_selection_list.load_pie,
                                                                    img_path,
                                                                    piechart_path),
                                                            -1)
                                        ## Select the image being processed
                                        for img_node in img_nodes:
                                            if img_name == img_node.file_name:
                                                Clock.schedule_once(
                                                    partial(self.list_box.img_selection_list.select_node,
                                                            img_node,
                                                            piechart_path),
                                                    0)
                                                break

                        if not self.input_box.tgl_colour_analysis.state == 'down':
                            # Sort the data based on date
                            single_array_healthy = dict(sorted(single_array_healthy.items()))
                            single_array_stress = dict(sorted(single_array_stress.items()))
                            
                            # Date adjustment based on correct screening start date
                            single_array_healthy_shifted = dict()
                            single_array_stress_shifted = dict()
                            ## Reference start date
                            if dataset_version == 1:
                                ref_date = datetime.strptime(self.screening_starts['DS_1'], "%Y-%m-%d")
                            else:
                                ref_date = datetime.strptime(self.screening_starts['DS_2'], "%Y-%m-%d")
                            
                            # Shift the healthy data
                            ## Find the deviation based on the first date 
                            date_delta = ref_date - datetime.strptime(list(single_array_healthy.keys())[0], "%Y-%m-%d")
                            for date_str in single_array_healthy:
                                date = datetime.strptime(date_str, "%Y-%m-%d")
                                ## Shifting the date
                                date += date_delta
                                single_array_healthy_shifted[datetime.strftime(date, "%Y-%m-%d")] \
                                    = single_array_healthy[date_str]

                            # Shift the stress data
                            ## Find the deviation based on the first date 
                            date_delta = ref_date - datetime.strptime(list(single_array_stress.keys())[0], "%Y-%m-%d")
                            for date_str in single_array_stress:
                                date = datetime.strptime(date_str, "%Y-%m-%d")
                                ## Shifting the date
                                date += date_delta
                                single_array_stress_shifted[datetime.strftime(date, "%Y-%m-%d")] \
                                    = single_array_stress[date_str]

                            all_array_healthy[rep_name] = single_array_healthy_shifted
                            all_array_stress[rep_name] = single_array_stress_shifted
                
                if not self.input_box.tgl_colour_analysis.state == 'down':
                    
                    save_plot_numbers(
                        os.path.join(output_root_dir, class_name, 'plots'),
                        all_array_stress,
                        all_array_healthy,
                    )

                    # Stress area plotting
                    try:
                        plotting(
                            all_array_stress,
                            os.path.join(output_root_dir, class_name, 'plots'),
                            dataset_version=dataset_version,
                            class_name=class_name,
                            data_type="Stress",
                        )
                    except Exception as e:
                        print(f'{e}: all_array_stress maybe empty')

                    # Healthy area plotting
                    try:
                        plotting(
                            all_array_healthy,
                            os.path.join(output_root_dir, class_name, 'plots'),
                            dataset_version=dataset_version,
                            class_name=class_name,
                            data_type="Healthy",
                        )
                    except Exception as e:
                        print(f'{e}: all_array_healthy maybe empty')

        self.enable_controls()
        self.input_box.enable_controls()
        Clock.schedule_once(self.reset_selection, 0)

        print(f"Finished at {datetime.now().time()}")
        self.is_processing = False


    def enable_controls(self):
        self.chk_segmented.disabled = False
        self.chk_mask.disabled = False
        self.chk_stress.disabled = False
        self.chk_selected_reps.disabled = False
        self.txt_selected_reps.disabled = False
        self.chk_selected_eco.disabled = False
        self.chk_all_classes.disabled = False
        self.btn_sow_date.disabled = False
        self.btn_screening_date.disabled = False
        self.btn_experiment_end_date.disabled = False
        self.btn_reset.disabled = False
        self.btn_reset_selection.disabled = False
        self.btn_process.disabled = False

    def disable_controls(self):
        self.chk_segmented.disabled = True
        self.chk_mask.disabled = True
        self.chk_stress.disabled = True
        self.chk_selected_reps.disabled = True
        self.txt_selected_reps.disabled = True
        self.chk_selected_eco.disabled = True
        self.chk_all_classes.disabled = True
        self.btn_sow_date.disabled = True
        self.btn_screening_date.disabled = True
        self.btn_experiment_end_date.disabled = True
        self.btn_reset.disabled = True
        self.btn_reset_selection.disabled = True
        self.btn_process.disabled = True


class CustomCheckBox(CheckBox):
    def on_touch_down(self, *args):
        if self.active:
            return
        super(CustomCheckBox, self).on_touch_down(*args)
