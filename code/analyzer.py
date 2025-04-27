import os
# import io
# import shutil
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
# from tkcalendar import DateEntry
import matplotlib
matplotlib.use('agg')  # Non-interactive backend that can only write to files
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
from collections import Counter
from pathlib import Path
from datetime import datetime
import itertools


# Dataset specific data
calibration_factors = {"DS_1": 0.13715, "DS_2": 0.14690}
screening_starts = {"DS_1": "2022-05-11", "DS_2": "2022-08-02"}
screening_das = {"DS_1": 13, "DS_2": 11}


custom_cmap = [
    [0.96,	0.24,	0.05],
    [0.19,	0.22,	0.98],
    [0.92,	0.98,	0.15],
    [0.09,	0.76,	0.25],
    [0.96,	0.51,	0.19],
    [0.91,	0.31,	0.86],
    [0.00,	0.71,	0.68],
    [0.45,	0.00,	0.85],
    [0.75,	0.94,	0.27],
    [0.88,	0.38,	0.45],
    [0.98,	0.75,	0.83],
    [0.26,	0.83,	0.96],
    [0.61,	0.39,	0.14],
    [0.86,	0.75,	1.00],
    [0.27,	0.62,	0.86],
    [1.00,	0.85,	0.69],
    [0.38,	0.01,	0.15],
    [0.89,	0.84,	0.05],
    [0.31,	0.62,	0.33],
    [0.67,	0.09,	0.40],
    [0.67,	1.00,	0.76],
    [0.66,	0.66,	0.66],
    [0.71,	0.44,	0.47],
    [0.56,	0.47,	0.67],
    [0.04,	0.49,	0.96],
    [0.72,	0.27,	0.12],
    [0.60,	0.14,	0.96],
    [0.90,	0.88,	0.93],
    [0.55,	1.00,	0.32],
    [0.12,	0.78,	0.82],
    [0.76,	0.85,	0.41],
    [0.36,	0.08,	0.49],
    [0.71,	0.86,	0.67],
    [0.15,	0.01,	0.76],
    ]


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def percentage(part, whole):
    return "{:.2f}".format(float(part) / float(whole))


def map_to_standard_colors(rgb_colors, color_labeler):
    mapped_colors = []
    for rgb in rgb_colors:
        lab_color = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0]
        color_name = color_labeler.label_c(lab_color)
        mapped_colors.append(color_name)
    return mapped_colors


def color_region(image, num_clusters):
    (h, w) = image.shape[:2]
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image_RGB.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, (centers) = cv2.kmeans(
        pixel_values, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    labels_flat = labels.flatten()
    segmented_image = centers[labels_flat]
    segmented_image = segmented_image.reshape(image_RGB.shape)
    counts = Counter(labels_flat)
    counts = dict(sorted(counts.items()))
    center_colors = centers
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [np.array(ordered_colors[i]).reshape(1, 3) for i in counts.keys()]
    index_bkg = [
        index for index in range(len(hex_colors)) if hex_colors[index] == "#000000"
    ]
    if len(index_bkg) > 0:
        del hex_colors[index_bkg[0]]
        del rgb_colors[index_bkg[0]]
        delete = [key for key in counts if key == index_bkg[0]]
        for key in delete:
            del counts[key]
    list_counts = list(counts.values())
    color_ratio = [
        percentage(value_counts, np.sum(list_counts)) for value_counts in list_counts
    ]
    return rgb_colors, counts, hex_colors, color_ratio


def process_piecharts(image_path, save_path=None, folder_path=None, force_create=True):
    
    output_path = image_path.replace(os.path.abspath(folder_path),
                                     os.path.abspath(save_path))
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
        if force_create:
            # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
            image = cv2.imread('\\\\?\\' + os.path.abspath(image_path))
            num_cluster = 9
            (_, _, hex_colors, color_ratio) = color_region(image, num_cluster)
            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            fig, ax = plt.subplots(figsize=(531*px, 525*px))
            patches, texts = ax.pie(color_ratio, labels=hex_colors, colors=hex_colors)
            for txt in texts:
                txt.set_fontsize(14)
            # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
            os.makedirs(os.path.dirname('\\\\?\\' + os.path.abspath(output_path)), exist_ok=True)
            ax.figure.savefig('\\\\?\\' + os.path.abspath(output_path))
            plt.close()
            return output_path
        else:
            return None
    else:
        return output_path


# Function to convert date to DAS format
def date_to_das(date, start_date, dataset_version=1):
    date = datetime.strptime(date, "%Y-%m-%d")
    if dataset_version == 1:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        das = (date - start_date).days + screening_das['DS_1']  # Start DAS from 13
    else:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        das = (date - start_date).days + screening_das['DS_2']  # Start DAS from 11
    return f"{das}"


def update_dict_with_average(a, key, new_value):
    key = key[:10]
    if key in a:
        current_value = a[key]
        averaged_value = current_value + new_value
        a[key] = averaged_value / 2.0
    else:
        a[key] = new_value


def plotting(data, save_path, dataset_version=1, class_name=None, data_type="Stress"):
    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
    os.makedirs('\\\\?\\' + os.path.abspath(save_path), exist_ok=True)

    # Sort the dictionary by dates within each leaf's data
    data = {leaf: dict(sorted(data[leaf].items())) for leaf in data}

    # Determine the earliest date to use as the start date
    all_dates = [date for leaf_data in data.values() for date in leaf_data.keys()]
    start_date = min(all_dates)

    # Extract all unique dates
    dates = sorted(
        set(date for leaf_data in data.values() for date in leaf_data.keys())
    )

    # Convert dates to DAS format and sort them
    das_dates_mapping = {
        str(date).split(" ")[0]: date_to_das(date, start_date, dataset_version=dataset_version) for date in dates
    }

    leaves = list(data.keys())

    # Prepare data for plotting
    lengths = {
        leaf: [data[leaf].get(date, [None]) for date in dates] for leaf in leaves
    }

    # Get a colormap with a number of colors equal to the number of leaves
    if dataset_version == 1:
        cmap = plt.get_cmap(
            "tab20", len(leaves)
        )  # 'tab20' is just an example; you can use any colormap
    else:
        cmap = ListedColormap(custom_cmap)

    # Plotting
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(1200*px, 500*px), num="my_figure")

    lines = []
    labels = []

    for i, leaf in enumerate(leaves):
        
        # Extract lengths and corresponding dates where data is available
        available_dates = [
            date
            for date, length in zip(dates, lengths[leaf])
            if length is not None and length != [None]
        ]
        available_lengths = [
            length
            for length in lengths[leaf]
            if length is not None and length != [None]
        ]
        if all(x == 0 for x in available_lengths):
            # all lengths are 0. Continue to the next leaves
            continue

        # Data thresholding
        
        # Dataset 1
        if dataset_version == 1:
            
            '''[23-09-24] Requested to change the threshold for some reps on dataset 1'''
            if not class_name in ['Col-0', 'Can-0', 'Hovdala-2', 'Hs-0', 'Hsm', 'PHW-2', 'Wil-2']:
                # Length threshold value
                threshold = 0.02 * np.max(np.array(available_lengths))
                # Filtering the available lengths from leading zeros
                for idx, x in enumerate(available_lengths):
                    if x <= threshold:
                        # Set this data to nan only if the next data is still below the threshold
                        try:
                            if available_lengths[idx+1] <= threshold: 
                                available_lengths[idx] = float('nan')
                            else:
                                break
                        except:
                            pass
        
            else:
                # Length threshold value
                threshold = 0.03 * np.max(np.array(available_lengths))
                for idx, x in enumerate(available_lengths):
                    if x <= threshold:
                        # Set this data to nan only if the next data is still below the threshold
                        try:
                            # Set this data to nan only if the previous and next data is still below the threshold
                            if idx > 0 and (available_lengths[idx-1] > threshold):
                                continue
                            # This is the last data and previous data is below the threshold
                            if idx == len(available_lengths)-1:
                                available_lengths[idx] = float('nan')
                                continue
                            if available_lengths[idx+1] <= threshold:
                                available_lengths[idx] = float('nan')

                        except Exception as e:
                            print (e)
                            pass
        # Dataset 2
        else:
            
            if data_type == 'Healthy':
                # Length threshold value
                '''[12-12-24] Requested to change the threshold for Zdr-1 on dataset 2 due to lag start'''
                if class_name in ['Zdr-1']:
                    threshold = 0
                else:
                    threshold = 0.01 * np.max(np.array(available_lengths))
                for idx, x in enumerate(available_lengths):
                    if x <= threshold:
                        # Set this data to nan only if the next data is still below the threshold
                        try:
                            # Set this data to nan only if the previous and next data is still below the threshold
                            if idx > 0 and (available_lengths[idx-1] > threshold):
                                continue
                            # This is the last data and previous data is below the threshold
                            if idx == len(available_lengths)-1:
                                available_lengths[idx] = float('nan')
                                continue
                            if available_lengths[idx+1] <= threshold:
                                available_lengths[idx] = float('nan')

                        except Exception as e:
                            print (e)
                            pass
            
            if data_type == 'Stress':
                # Length threshold value
                threshold = 0.03 * np.max(np.array(available_lengths))
                for idx, x in enumerate(available_lengths):
                    if x <= threshold:
                        # Set this data to nan only if the next data is still below the threshold
                        try:
                            # Set this data to nan only if the previous and next data is still below the threshold
                            if idx > 0 and (available_lengths[idx-1] > threshold):
                                continue
                            # This is the last data and previous data is below the threshold
                            if idx == len(available_lengths)-1:
                                available_lengths[idx] = float('nan')
                                continue
                            if available_lengths[idx+1] <= threshold:
                                available_lengths[idx] = float('nan')

                        except Exception as e:
                            print (e)
                            pass

        (line,) = ax.plot(available_dates, 
                          available_lengths, 
                          marker=".", 
                          label=leaf, 
                          color=cmap(i))
        lines.append(line)
        labels.append(leaf)

    
    # If no data is plotted. Close the plot and return
    if lines == []:
        plt.close("my_figure")
        return
    
    ax.set_xlabel("Days After Sowing (DAS)")
    if data_type.lower()== 'stress':
        y_label_str = 'Stressed Rosette'
    else:
        y_label_str = 'Healthy Rosette'

    ax.set_ylabel("{} Area [mm\u00b2]".format(y_label_str))
    ax.set_title("{}".format(class_name))

    # Sort the legend handles and labels based on the numeric part of the labels
    sorted_lines_labels = sorted(
        zip(lines, labels), key=lambda x: int(x[1].split("_")[1])
    )
    sorted_lines, sorted_labels = zip(*sorted_lines_labels)

    if dataset_version == 1:
        ncol = 5
    else:
        ncol = 8
    sorted_lines = list(itertools.chain(*[sorted_lines[i::ncol] for i in range(ncol)]))
    sorted_labels = list(itertools.chain(*[sorted_labels[i::ncol] for i in range(ncol)]))

    # Place legend outside of the plot at the bottom
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Create legend with sorted handles and labels
    legend = ax.legend(
        sorted_lines,
        sorted_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        fancybox=True,
        shadow=True,
        ncol=ncol,
    )

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # After plotting
    # Get the current labels and their positions
    # Force the plot to update
    fig.canvas.draw()
    current_labels = ax.get_xticklabels()
    current_positions = ax.get_xticks()

    # Get the new labels in the correct order
    new_labels = []
    for label in current_labels:
        new_labels.append(das_dates_mapping[label.get_text()])

    # Set the new labels
    new_label_range = [
        str(i) for i in range(int(new_labels[0]), int(new_labels[-1]) + 1, 2)
    ]
    new_positions = np.array(
        [
            float(i)
            for i in range(int(current_positions[0]), int(current_positions[-1]) + 1, 2)
        ]
    )
    ax.set_xticklabels(new_label_range)

    # Optionally, you might need to adjust the positions if the number of new labels is different from the current ones
    ax.set_xticks(new_positions)

    # Save the figure
    file_name = os.path.join(save_path, "{}_by_date.png".format(data_type))
    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
    fig.savefig(
        '\\\\?\\' + os.path.abspath(file_name),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )

    # Plot resize and padding
    target_size = (1020, 540)
    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
    img = cv2.imread('\\\\?\\' + os.path.abspath(file_name))
    factor_y = target_size[1] / img.shape[0]
    factor_x = target_size[0] / img.shape[1]
    factor = min (factor_x, factor_y)
    img = cv2.resize(img, (int(img.shape[1]* factor), int(img.shape[0]*factor)))
    ## Padding
    diff_y = target_size[1] - img.shape[0]
    diff_x = target_size[0] - img.shape[1]
    img = np.pad(img,((diff_y//2, diff_y - diff_y//2), (diff_x//2, diff_x-diff_x//2), (0,0)), 'edge')
    # Rewrite the image
    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
    cv2.imwrite('\\\\?\\' + os.path.abspath(file_name), img)

    plt.close("my_figure")


def extract_rep_name(rep_name):
    return rep_name.split("_")[-1]

def extract_timestamp_from_filename(filename):
    # Assuming the timestamp is always in this format: YYYY_MM_DD_HH_MM_SS
    parts = filename.split("segmented_images" + os.sep)[1]
    date_time_str = parts.split("-")[0][8:]
    return datetime.strptime(date_time_str[-19:], "%Y_%m_%d_%H_%M_%S")


def extract_timestamp_from_filename_only(filename):
    date_time_str = filename.split("-")[0][8:]
    if date_time_str[0] == "_":
        date_time_str = date_time_str[1:]
    return datetime.strptime(date_time_str, "%Y_%m_%d_%H_%M_%S")


# Function to convert RGB to HSV
def rgb_hsv(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)



def bgr_hsv(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)


# Function to convert BGR to RGB
def rgb_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Function to segment the image based on the number of clusters
def color_cluster_seg(image, num_clusters, random_state=2024):
    reshaped = image.reshape(-1, 3)
    kmeans = KMeans(
        n_clusters=num_clusters,
        init="random",
        n_init=10,
        max_iter=100,
        random_state=random_state,
    ).fit(reshaped)
    clustering = kmeans.labels_.reshape(image.shape[:2])
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)

    for i in range(num_clusters):
        kmeansImage[clustering == i] = int(255 / (num_clusters - 1)) * i

    return kmeansImage


# Function to detect the healthy green areas based on HSV thresholds
def img_segmentation(hsv_img, lower_green, upper_green):
    return cv2.inRange(hsv_img, lower_green, upper_green)


# Function to process the contours and highlight stressed areas
def process_contours(img_real, img_mask, healthy_mask, dataset_version=1):
    contours_mask, hierarchy = cv2.findContours(
        img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    sorted_contours = sorted(contours_mask, key=cv2.contourArea, reverse=True)

    # Convert to grayscale
    gray_image = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)

    # Calculate the total number of non-zero pixels
    total_area = cv2.countNonZero(gray_image)

    if len(sorted_contours) > 0:
        max_area = int(cv2.contourArea(sorted_contours[0]))
        if max_area > 6500:
            center = cv2.moments(sorted_contours[0])
            cX = int(center["m10"] / center["m00"])
            cY = int(center["m01"] / center["m00"])
            cv2.circle(img_mask, (cX, cY), int(max_area / 1200), (0, 0, 0), -1)

            healthy_mask = cv2.dilate(
                healthy_mask, np.ones((2, 2), np.uint8), iterations=1
            )
            healthy_mask_gray = contour_size(healthy_mask)
            contours_mask, hierarchy = cv2.findContours(
                healthy_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.circle(
                healthy_mask_gray, (cX, cY), int(max_area / 1200), (0, 0, 0), -1
            )

            out = cv2.bitwise_xor(img_mask, healthy_mask_gray)
            out = cv2.erode(out, np.ones((2, 3), np.uint8))
            out = cv2.dilate(out, np.ones((2, 2), np.uint8))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            contours, hierarchy = cv2.findContours(
                out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour_list = [
                contour
                for contour in contours
                if cv2.contourArea(contour) >= max_area / 110
            ]

            stressed_area = 0
            for contour in contour_list:
                stressed_area += cv2.contourArea(contour)
            healthy_area = total_area - stressed_area
            if contour_list:
                cv2.drawContours(img_real, contour_list, -1, (0, 0, 255), 2)
        else:
            stressed_area = 0
            healthy_area = total_area
    
    # Determine cal factor
    if dataset_version==1:
        # Dataset version 1
        cal_factor = calibration_factors['DS_1']
    else:
        # Dataset version 2
        cal_factor = calibration_factors['DS_2']

    return img_real, stressed_area * cal_factor * cal_factor, healthy_area * cal_factor * cal_factor


# Function to detect the contour size
def contour_size(image):
    if len(image.shape) != 2:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), connectivity=8
        )
    else:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            image, connectivity=8
        )

    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    min_size = 100
    max_size = image.shape[0] * image.shape[1] * 4

    connected = np.zeros(image.shape, dtype=np.uint8)
    for i in range(nb_components):
        if min_size <= sizes[i] < max_size:
            connected[output == i + 1] = 255

    return connected


def update_image(img, num_clusters, lower_green, upper_green, dataset_version=1):

    if dataset_version == 1:
        img = np.array(img)
        rgb = rgb_bgr(np.array(img))  # BGR
        hsv = bgr_hsv(rgb)
        healthy_mask = img_segmentation(hsv, lower_green, upper_green)
        max_area = 0.0
        random_state = 2024

        while True:

            clustered_img = color_cluster_seg(rgb, num_clusters, random_state=random_state)
            contours_mask, _ = cv2.findContours(clustered_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours_mask, key=cv2.contourArea, reverse=True)

            if len(sorted_contours) > 0:
                max_area = int(cv2.contourArea(sorted_contours[0]))
            if max_area < 250000:
                break
            random_state += 1
        output_img, stressed_area, healthy_area = process_contours(rgb.copy(), clustered_img, healthy_mask, dataset_version)

        return Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)), stressed_area, healthy_area

    else:
        img = np.array(img)
        rgb = rgb_bgr(np.array(img))  # BGR
        hsv = bgr_hsv(rgb)
        healthy_mask = img_segmentation(hsv, lower_green, upper_green)
        random_state = 2024

        while True:

            max_area = 0.0
            clustered_img = color_cluster_seg(rgb, num_clusters, random_state=random_state)
            contours_mask, _ = cv2.findContours(clustered_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours_mask, key=cv2.contourArea, reverse=True)

            if len(sorted_contours) > 0:
                for i in sorted_contours:
                    max_area += int(cv2.contourArea(i))
            if max_area < ((img.shape[0] - 0) * (img.shape[1] - 0))/2:
                break
            random_state += 1
            
        output_img, stressed_area, healthy_area = process_contours(rgb.copy(), clustered_img, healthy_mask, dataset_version)

        return Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)), stressed_area, healthy_area


# Function to save the slider values to a text file
def save_slider_values(output_path, num_clusters, lower_green, upper_green):
    txt_file_path = output_path.replace("stressed_area", "text")
    txt_file_path = txt_file_path.replace("_seg.png", "_seg.txt")

    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
    os.makedirs('\\\\?\\' + os.path.abspath(os.path.dirname(txt_file_path)), exist_ok=True)

    with open(txt_file_path, "w") as f:
        f.write(f"Clusters: {num_clusters}\n")
        f.write(f"Lower Hue: {lower_green[0]}\n")
        f.write(f"Lower Saturation: {lower_green[1]}\n")
        f.write(f"Lower Value: {lower_green[2]}\n")
        f.write(f"Upper Hue: {upper_green[0]}\n")
        f.write(f"Upper Saturation: {upper_green[1]}\n")
        f.write(f"Upper Value: {upper_green[2]}\n")


def save_plot_numbers(output_path, array_stress, array_healthy):
    json_file_path_stress = os.path.join(output_path, "plot_stress.json")

    json_file_path_healthy = os.path.join(output_path, "plot_healthy.json")

    # We add '\\\\?\\' to avoid bumping into Windows' path length limitation to 260 characters
    os.makedirs('\\\\?\\' + os.path.abspath(os.path.dirname(json_file_path_stress)), exist_ok=True)

    with open(json_file_path_stress, "w") as fp:
        json.dump(array_stress, fp)

    with open(json_file_path_healthy, "w") as fp:
        json.dump(array_healthy, fp)


# Function to load slider values from a text file
def load_slider_values(image_path):
    original_path = Path(image_path)

    # Step 1: Replace the directory containing 'accessions_dataset1_labels'
    parts = list(original_path.parts)
    parts[-5] = parts[-5] + '_processed_texts'
    parts = [part for part in parts if part != "segmented_images"]
    txt_file_path = str(Path(*parts).with_suffix('.txt'))

    if os.path.exists(txt_file_path):
        with open(txt_file_path, "r") as f:
            lines = f.readlines()
            num_clusters = int(lines[0].split(": ")[1])
            lower_green = (
                int(lines[1].split(": ")[1]),
                int(lines[2].split(": ")[1]),
                int(lines[3].split(": ")[1]),
            )
            upper_green = (
                int(lines[4].split(": ")[1]),
                int(lines[5].split(": ")[1]),
                int(lines[6].split(": ")[1]),
            )
            return num_clusters, lower_green, upper_green
    else:
        # Default values if the text file doesn't exist
        return 2, (30, 70, 50), (95, 255, 110)
