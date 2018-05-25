import argparse
import os
import sys
import timeit
import json
from multiprocessing import Pool, cpu_count
from functools import partial
from pkg_resources import resource_filename

import numpy as np
import pandas as pd
from scipy import linalg, misc
from skimage import color
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


from .import hasel

# Optional imports of pandas and seaborn are located in functions
# group_analyze() and plot_group().


def parse_arguments():
    """
    Parsing arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to the directory or file")
    parser.add_argument("-t0", "--thresh0", required=False,
                        type=int, help="Global threshold for stain-positive area of channel_0 stain. "
                                       "Accepted values from 0 to 100.")
    parser.add_argument("-t1", "--thresh1", required=False,
                        type=int, help="Global threshold for stain-positive area of channel_1 stain. "
                                       "Accepted values from 0 to 100.")
    parser.add_argument("-e", "--empty", required=False, default=101,
                        type=int, help="Global threshold for EMPTY area,"
                                       "from 0 to 100. Default value is 101,"
                                       " which is equal to disabled empty area filter.")
    parser.add_argument("-s", "--silent", required=False, help="Supress figure rendering during the analysis,"
                                                               " only final results"
                                                               " will be saved", action="store_true")
    parser.add_argument("-a", "--analyze", required=False, help="Add group analysis after the indvidual image"
                                                                " processing. The groups are created using the"
                                                                " filename. Everything before '_' symbol will"
                                                                " be recognized as a group name. Example:"
                                                                " sample01_10.jpg, sample01_11.jpg will be"
                                                                " counted as a single group 'sample01'",
                                                                action="store_true")
    parser.add_argument("-d", "--dpi", required=False, default=200, type=int, help="Output images DPI. 900 is "
                                                                                   "recommended for printing quality."
                                                                                   " High resolution can significally"
                                                                                   " slow down the process.")
    parser.add_argument("-rs", "--resize", required=False, nargs='+', default=(768, 1024), type=int,
                        help="Image resolution for processing speed up. Higher resolution increases the accuracy,"
                             " but can significally slow down the process. Default value is (768,1024)."
                             " Pass values like '--resize 768 1024'")
    parser.add_argument("-m", "--matrix", required=False, help="Your matrix in a JSON formatted file")
    parser.add_argument("-sc", "--save_channels", required=False, help="Save separate stain channels to subfolder",
                        action="store_true")
    parser.add_argument("-n", "--notch", required=False, action="store_true",
                        help="Notches for boxplot in group analysis to show confidence interval")
    arguments = parser.parse_args()
    return arguments



def get_image_filenames(path):
    """
    Returns only the filenames in the path. Directories, subdirectories and files below the first level
    are excluded
    """

    return [name for name in sorted(os.listdir(path))
            if not os.path.isdir(os.path.join(path, name))]


def validate_input(filenames_raw):
    """
    Function is validating the input. Error is risen when filenames are empty. Non-graphical files are excluded.
    :param: filenames_raw
    :return: filenames_validated
    """
    extensions = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp', '.JPG', '.JPEG', '.TIF', '.TIFF', '.PNG', '.BMP')
    filenames_validated = [filename for filename in filenames_raw if filename.endswith(extensions)]
    if not filenames_validated:
        sys.exit("Error. No image files to analyse. Input directory is empty")

    return filenames_validated


def calc_deconv_matrix(vector_raw_stains):
    """
    Calculating matrix for deconvolution
    """

    vector_raw_stains[2, :] = np.cross(vector_raw_stains[0, :], vector_raw_stains[1, :])
    matrix_stains = linalg.inv(vector_raw_stains)
    return matrix_stains


def separate_channels(image_original, matrix_dh, args):
    """
    Separate the stains using the custom matrix
    """
    parsed_json = json_parse(args)
    # Histogram shift. This correction makes the background really blank. After the correction
    # numpy clipping is performed to fit the 0-100 range
    hist_shift_0 = parsed_json["hist_shift_0"]
    hist_shift_1 = parsed_json["hist_shift_1"]
    hist_shift_2 = parsed_json["hist_shift_2"]

    image_separated = color.separate_stains(image_original, matrix_dh)
    stain_ch0 = image_separated[..., 0]
    stain_ch1 = image_separated[..., 1]
    stain_ch2 = image_separated[..., 2]

    stain_ch0 = (stain_ch0 + 1) * 200
    stain_ch0 -= hist_shift_0
    stain_ch0 = np.clip(stain_ch0, 0, 100)

    stain_ch1 = (stain_ch1 + 1) * 200
    stain_ch1 -= hist_shift_1
    stain_ch1 = np.clip(stain_ch1, 0, 100)

    stain_ch2 = (stain_ch2 + 1) * 200
    stain_ch2 -= hist_shift_2
    stain_ch2 = np.clip(stain_ch2, 0, 100)

    # Extracting Lightness channel from HSL of original image
    # L-channel is multiplied to 100 to get the range 0-100 % from 0-1. It's easier to use with
    # empty area threshold
    image_hsl = hasel.rgb2hsl(image_original)
    channel_lightness = (image_hsl[..., 2] * 100)
    return stain_ch0, stain_ch1, stain_ch2, channel_lightness


def log_and_console(path_output_log, text_log, bool_log_new=False):
    """
    Write the log and show the text in console
    bool_log_new is used to erase the log file if it exists to avoid appending new data to the old one
    """

    if bool_log_new:
        print(text_log)
        # Initialize empty file
        with open(path_output_log, "a") as fileLog:
            fileLog.write("")
        with open(path_output_log, "w") as fileLog:
            fileLog.write(text_log)
            fileLog.write('\n')
    else:
        print(text_log)
        with open(path_output_log, "a") as fileLog:
            fileLog.write(text_log)
            fileLog.write('\n')


def log_only(path_output_log, text_log):
    """
    Write the log to the file only
    """
    with open(path_output_log, "a") as fileLog:
            fileLog.write(text_log)
            fileLog.write('\n')


def count_thresholds(stain, channel_lightness, thresh_channel, thresh_empty_default):
    """
    Counts thresholds. "stain" is a distribution map of stain, channel_lightness is a L channel from
    original image in HSL color space. The output are the thresholded images of stain-positive areas and
    empty areas.
    """

    thresh_stain = stain > thresh_channel
    thresh_empty = channel_lightness > thresh_empty_default
    return thresh_stain, thresh_empty


def count_areas(thresh_stain, thresh_empty):
    """
    Count areas from numpy arrays
    """

    area_all = float(thresh_stain.size)
    area_empty = float(np.count_nonzero(thresh_empty))
    area_stain_pos = float(np.count_nonzero(thresh_stain))

    # Count relative areas in % with rounding
    # NB! Relative stained area is counted without empty areas if active
    area_rel_empty = round((area_empty / area_all * 100), 2)
    area_rel_stain = round((area_stain_pos / (area_all - area_empty) * 100), 2)
    return area_rel_empty, area_rel_stain


def stack_data(list_filenames, list_data, args):
    """
    Function stacks the data from data lists.
    """
    parsed_json = json_parse(args)
    str_ch0 = parsed_json["channel_0"]
    str_ch1 = parsed_json["channel_1"]
    str_col0 = str_ch0 + "-positive area, %"
    str_col1 = str_ch1 + "-positive area, %"
    pandas_df = pd.DataFrame(data=list_data, columns=[str_col0, str_col1], index=list_filenames)
    pandas_df.index.name = 'Filename'
    return pandas_df


def save_data(path_output_csv, array_filenames, list_data, args):
    """
    Function puts data array to the output csv file.
    """
    data_output = stack_data(array_filenames, list_data, args)
    print(data_output)

    # write array to csv file
    data_output.to_csv(path_output_csv)


def get_output_paths(path_root):
    """
    Output path generating
    """

    path_output = os.path.join(path_root, "result/")
    path_output_log = os.path.join(path_output, "log.txt")
    path_output_csv = os.path.join(path_output, "analysis.csv")
    return path_output, path_output_log, path_output_csv


def check_mkdir_output_path(path_output):
    """
    Function checks if the output path exists and creates it if not
    """

    if not os.path.exists(path_output):
        os.mkdir(path_output)

def resize_input_image(image_original, size):
    """
    Resizing the original images makes the slowest functions calc_deconv_matrix() and hasel.hsl2rgb()
    work much faster. There are no visual troubles or negative effects to the accuracy.
    """

    image_original = misc.imresize(image_original, size, interp='nearest')
    return image_original


def image_process(var_pause, matrix_stains, path_output, pathOutputLog, str_ch0, str_ch1, str_ch2,
                  thresh_0, thresh_1, args, filename):
    """
    Main cycle, split into several processes using the Pool(). All images pass through this
    function. The result of this function is composite images, saved in the target directory,
    log output and array_data - numpy array, containing the data obtained.
    Optimized thresholds are loaded from json with stain parameters
    """

    path_input_image = os.path.join(args.path, filename)
    path_output_image = os.path.join(path_output, filename.split(".")[0] + "_analysis.png")
    image_original = mpimg.imread(path_input_image)

    resize_resolution = tuple(args.resize)
    image_original = resize_input_image(image_original, resize_resolution)

    stain_ch0, stain_ch1, stain_ch2, channel_lightness = separate_channels(image_original, matrix_stains, args)

    thresh_stain_ch0, thresh_empty = count_thresholds(stain_ch0, channel_lightness, thresh_0, args.empty)
    area_rel_empty, area_rel_stain_ch0 = count_areas(thresh_stain_ch0, thresh_empty)
    thresh_stain_ch1, thresh_empty = count_thresholds(stain_ch1, channel_lightness, thresh_1, args.empty)
    area_rel_empty, area_rel_stain_ch1 = count_areas(thresh_stain_ch1, thresh_empty)

    # Close all figures after cycle end
    plt.close('all')

    list_rel_area = ([area_rel_stain_ch0, area_rel_stain_ch1])

    # Optional. Save the separate channels of used stains
    if args.save_channels:
        path_channel_subdir = os.path.join(path_output, "separate_channels/")
        check_mkdir_output_path(path_channel_subdir)
        plot_channels(filename, stain_ch0, path_channel_subdir, pathOutputLog, str_ch0, args.dpi)
        plot_channels(filename, stain_ch1, path_channel_subdir, pathOutputLog, str_ch1, args.dpi)
        plot_channels(filename, thresh_stain_ch0, path_channel_subdir, pathOutputLog, str_ch0 + '-positive area', args.dpi)
        plot_channels(filename, thresh_stain_ch1, path_channel_subdir, pathOutputLog, str_ch1 + '-positive area', args.dpi)

    # Creating the composite image
    plot_reference_figure(image_original, stain_ch0, stain_ch1, stain_ch2, thresh_stain_ch0, thresh_stain_ch1,
                          str_ch0, str_ch1, str_ch2)
    plt.savefig(path_output_image, dpi=args.dpi)

    log_and_console(pathOutputLog, "Image saved: {}".format(path_output_image))

    # In silent mode image will be closed immediately
    if not args.silent:
        plt.pause(var_pause)

    return list_rel_area


def group_filenames(filenames):
    """
    Creates groups of samples cutting filename after "_" symbol. Used to count
     statistics for each sample group.
    """

    list_file_group = []
    for filename in filenames:
        filename = filename.split("_")[0]
        list_file_group.append(filename)
    return list_file_group


def group_analyze(filenames, list_data, str_ch0, str_ch1, path_output, args):
    """
    Statistical group analysis. Output is csv file with results, and group plot
    function.
    """

    # Creating groups of samples using the filename
    list_file_group = group_filenames(filenames)

    str_col0 = str_ch0 + "-positive area, %"
    str_col1 = str_ch1 + "-positive area, %"

    # Creating pandas DataFrame
    column_names = [str_col0, str_col1]
    df = pd.DataFrame(list_data, columns=column_names, index=list_file_group)
    df.index.name = 'Group'
    df = df.apply(pd.to_numeric, errors='ignore')

    # Counting unique group number. Would be used for plot horizontal aspect
    group_num = df.groupby(df.index)[str_col0].nunique()
    group_num = len(group_num.index)

    # Channel_0 stats
    groupby_ch0 = df[str_col0].groupby(df.index)
    data_stats = groupby_ch0.agg([np.mean, np.std, np.median, np.min, np.max])
    path_output_stats = os.path.join(path_output, "group_stats_" + str_ch0 + ".csv")
    data_stats.to_csv(path_output_stats)
    print(data_stats)
    plot_group(df, path_output, str_ch0, str_col0, args, group_num)

    # Channel_1 stats
    groupby_ch1 = df[str_col1].groupby(df.index)
    data_stats = groupby_ch1.agg([np.mean, np.std, np.median, np.min, np.max])
    path_output_stats = os.path.join(path_output, "group_stats_" + str_ch1 + ".csv")
    data_stats.to_csv(path_output_stats)
    print(data_stats)
    plot_group(df, path_output, str_ch1, str_col1, args, group_num)


def plot_reference_figure(image_original, stain_ch0, stain_ch1, stain_ch2, thresh_stain_ch0, thresh_stain_ch1,
                          str_ch0, str_ch1, str_ch2):
    """
    Function plots the figure for every sample image.
    """
    plt.figure(num=None, figsize=(14, 7), dpi=150, facecolor='w', edgecolor='k')
    plt.subplot(231)
    plt.title('Original')
    plt.imshow(image_original)

    plt.subplot(232)
    plt.title(str_ch0)
    plt.imshow(stain_ch0, cmap=plt.cm.gray)

    plt.subplot(233)
    plt.title(str_ch1)
    plt.imshow(stain_ch1, cmap=plt.cm.gray)

    plt.subplot(234)
    plt.title(str_ch2)
    plt.imshow(stain_ch2, cmap=plt.cm.gray)

    plt.subplot(235)
    plt.title(str_ch0 + '-positive area')
    plt.imshow(thresh_stain_ch0, cmap=plt.cm.gray)

    plt.subplot(236)
    plt.title(str_ch1 + '-positive area')
    plt.imshow(thresh_stain_ch1, cmap=plt.cm.gray)

    plt.tight_layout()


def plot_group(data_frame, path_output, str_ch, str_col, args, group_num):
    # optional import
    import seaborn as sns

    path_output_image_png = os.path.join(path_output, "group_stats_" + str_ch + ".png")
    path_output_image_svg = os.path.join(path_output, "group_stats_" + str_ch + ".svg")
    print(data_frame)
    sns.set_style("whitegrid")
    sns.set_context("talk")
    # figsize aspect is regulated by number of groups to match the cases with
    # small and large group number
    plt.figure(num=None, figsize=(2*group_num, 7), dpi=150)
    plt.ylim(0, 100)
    sns.boxplot(x=data_frame.index, y=str_col, data=data_frame, notch=args.notch)
    plt.tight_layout()
    plt.savefig(path_output_image_png, dpi=args.dpi)
    plt.savefig(path_output_image_svg)


def plot_channels(filename, channel, path_channel_subdir, path_output_log, str_ch, dpi):
    path_output_image = os.path.join(path_channel_subdir, str_ch + "_channel_" + filename.split(".")[0] +".png")
    misc.imsave(path_output_image, channel)
    log_and_console(path_output_log, "Image saved: {}".format(path_output_image))


def json_parse(args):

    if args.matrix:
        json_path = args.matrix
    else:
        json_path = resource_filename(__name__, 'resources/dab.json')

    json_data = open(json_path)
    parsed_json = json.load(json_data)
    return parsed_json


def main():
    """
    """
    # Parse the arguments
    args = parse_arguments()

    list_data = []
    # Pause in seconds between the composite images when --silent(-s) argument is not active
    var_pause = 5
    # Initialize the global timer
    start_time_global = timeit.default_timer()

    # load internal resources in json format
    parsed_json = json_parse(args)

    vector_raw_stain = np.array(parsed_json["vector"])
    str_ch0 = parsed_json["channel_0"]
    str_ch1 = parsed_json["channel_1"]
    str_ch2 = parsed_json["channel_2"]

    path_output, path_output_log, path_output_csv = get_output_paths(args.path)

    check_mkdir_output_path(path_output)
    filenames = get_image_filenames(args.path)
    filenames = validate_input(filenames)

    # Thresholds are got from predefined values in JSON for selected stain or from user-defined args
    if args.thresh0:
        thresh_0 = args.thresh0
    else:
        thresh_0 = parsed_json["thresh_0"]

    if args.thresh1:
        thresh_1 = args.thresh1
    else:
        thresh_1 = parsed_json["thresh_1"]

    log_and_console(path_output_log, "Images for analysis: " + str(len(filenames)), True)
    log_and_console(path_output_log, "{} threshold = {}, {} threshold = {},"
                                     " Empty threshold = {}".format(str_ch0, thresh_0, str_ch1,
                                                                    thresh_1, args.empty))

    if args.empty > 100:
        log_and_console(path_output_log, "Empty area filtering is disabled.")

    # Calculate the stain deconvolution matrix
    matrix_stains = calc_deconv_matrix(vector_raw_stain)

    # Multiprocess implementation
    cores = cpu_count()
    log_and_console(path_output_log, "CPU cores used: {}".format(cores))

    # Main cycle where the images are processed and the data is obtained
    pool = Pool(cores)

    wrapper_image_process = partial(image_process, var_pause, matrix_stains,
                                    path_output, path_output_log, str_ch0, str_ch1, str_ch2, thresh_0, thresh_1, args)
    for poolResult in pool.imap(wrapper_image_process, filenames):
        list_data.append(poolResult)
    pool.close()
    pool.join()

    list_filenames = []
    for filename in filenames:
        list_filenames.append(filename)

    # Creating summary csv after main cycle end
    save_data(path_output_csv, list_filenames, list_data, args)

    # Optional statistical group analysis.
    if args.analyze:
        log_and_console(path_output_log, "Group analysis is active")
        group_analyze(filenames, list_data, str_ch0, str_ch1, path_output, args)

    # End of the global timer
    elapsed_global = timeit.default_timer() - start_time_global
    if not args.silent:
        average_image_time = (elapsed_global - len(filenames)*var_pause)/len(filenames)  # compensate the pause
    else:
        average_image_time = elapsed_global/len(filenames)
    log_and_console(path_output_log, "Analysis time: {:.1f} seconds".format(elapsed_global))
    log_and_console(path_output_log, "Average time per image: {:.1f} seconds".format(average_image_time))
