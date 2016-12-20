import argparse
import os
import csv
import timeit
import json
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
from scipy import linalg, misc
from skimage import color
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


from .import hasel
from .import resources

# Optional imports of pandas and seaborn are located in functions
# group_analyze() and plot_group().


def parse_arguments():
    """
    Parsing arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to the directory or file")
    parser.add_argument("-t0", "--thresh0", required=False, default=30,
                        type=int, help="Global threshold for stain-positive area of channel_0 stain. "
                                       "Accepted values from 0 to 100.")
    parser.add_argument("-t1", "--thresh1", required=False, default=40,
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
    arguments = parser.parse_args()
    return arguments


def get_image_filenames(path):
    """
    Returns only the filenames in the path. Directories, subdirectories and files below the first level
    are excluded
    """

    return [name for name in sorted(os.listdir(path))
            if not os.path.isdir(os.path.join(path, name))]


def calc_deconv_matrix(vector_raw_stains):
    """
    Calculating matrix for deconvolution
    """

    vector_raw_stains[2, :] = np.cross(vector_raw_stains[0, :], vector_raw_stains[1, :])
    matrix_stains = linalg.inv(vector_raw_stains)
    return matrix_stains


def separate_channels(image_original, matrix_dh):
    """
    Separate the stains using the custom matrix
    """

    image_separated = color.separate_stains(image_original, matrix_dh)
    stain_ch0 = image_separated[..., 0]
    stain_ch1 = image_separated[..., 1]
    stain_ch2 = image_separated[..., 2]

    # todo: move pre-contrast settings to json external file. Optimized values should be imported with stain matrix
    stain_ch0 = (stain_ch0 + 1) * 200
    # Histogram shift. This correction makes the background really blank. After the correction
    stain_ch0 -= 5
    # numpy clipping is performed to fit the 0-100 range
    stain_ch0 = np.clip(stain_ch0, 0, 100)

    stain_ch1 = (stain_ch1 + 1) * 200
    stain_ch1 -= 18
    stain_ch1 = np.clip(stain_ch1, 0, 100)

    stain_ch2 = (stain_ch2 + 1) * 200
    stain_ch2 -= 0
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


def stack_data(list_filenames, list_data):
    """
    Function stacks the data from data lists.
    """
    matrix_json = resources.import_vector()
    parsedJSON = json.loads(matrix_json)
    str_ch0 = parsedJSON["channel_0"]
    str_ch1 = parsedJSON["channel_1"]
    str_col0 = str_ch0 + "-positive area, %"
    str_col1 = str_ch1 + "-positive area, %"
    pandas_df = pd.DataFrame(data = list_data, columns=[str_col0, str_col1], index = list_filenames)
    return pandas_df


def save_csv(path_output_csv, array_filenames, list_data):
    """
    Function puts data array to the output csv file.
    """
    data_output = stack_data(array_filenames, list_data)
    print(data_output)

    # write array to csv file
    data_output.to_csv(path_output_csv)
    print("CSV saved: " + path_output_csv)


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
        print("Created result directory")
    else:
        print("Output result directory already exists. All the files inside would be overwritten!")


def resize_input_image(image_original, size):
    """
    Resizing the original images makes the slowest functions calc_deconv_matrix() and hasel.hsl2rgb()
    work much faster. There are no visual troubles or negative effects to the accuracy.
    """

    image_original = misc.imresize(image_original, size, interp='nearest')
    return image_original


def image_process(var_pause, matrix_stains, args, pathOutput, pathOutputLog, str_ch0, str_ch1, str_ch2,
                  filename):
    """
    Main cycle, split into several processes using the Pool(). All images pass through this
    function. The result of this function is composite images, saved in the target directory,
    log output and array_data - numpy array, containing the data obtained.
    """

    path_input_image = os.path.join(args.path, filename)
    path_output_image = os.path.join(pathOutput, filename.split(".")[0] + "_analysis.png")
    image_original = mpimg.imread(path_input_image)

    size_image = 480, 640
    image_original = resize_input_image(image_original, size_image)

    stain_ch0, stain_ch1, stain_ch2, channel_lightness = separate_channels(image_original, matrix_stains)

    thresh_stain_ch0, thresh_empty = count_thresholds(stain_ch0, channel_lightness, args.thresh0, args.empty)
    area_rel_empty, area_rel_stain_ch0 = count_areas(thresh_stain_ch0, thresh_empty)

    thresh_stain_ch1, thresh_empty = count_thresholds(stain_ch1, channel_lightness, args.thresh1, args.empty)
    area_rel_empty, area_rel_stain_ch1 = count_areas(thresh_stain_ch1, thresh_empty)

    # Close all figures after cycle end
    plt.close('all')

    list_rel_area = ([area_rel_stain_ch0, area_rel_stain_ch1])


    # Creating the complex image
    plot_figure(image_original, stain_ch0, stain_ch1, stain_ch2, channel_lightness, thresh_stain_ch0, thresh_stain_ch1,
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

    array_file_group = np.empty([0, 1])
    for filename in filenames:
        filename = filename.split("_")[0]
        array_file_group = np.vstack((array_file_group, filename))
    return array_file_group


def group_analyze(filenames, array_data, path_output, dpi):
    path_output_stats = os.path.join(path_output, "stats.csv")
    # Creating groups of samples using the filename
    array_file_group = group_filenames(filenames)

    # Creating pandas DataFrame
    column_names = ['Group', 'Stain+ area, %']
    data_frame_data = np.hstack((array_file_group, array_data))

    df = pd.DataFrame(data_frame_data, columns=column_names)
    df = df.apply(pd.to_numeric, errors='ignore')
    groupby_group = df['Stain+ area, %'].groupby(df['Group'])
    data_stats = groupby_group.agg([np.mean, np.std, np.median, np.min, np.max])
    data_stats.to_csv(path_output_stats)
    print(data_stats)
    plot_group(df, path_output, dpi)


def plot_figure(image_original, stain_ch0, stain_ch1, stain_ch2, channel_lightness, thresh_stain_ch0, thresh_stain_ch1,
                str_ch0, str_ch1, str_ch2):
    """
    Function plots the figure for every sample image. It creates the histogram from the stain array.
    Then it takes the bins values and clears the plot. That's done because fill_between function doesn't
    work with histogram but only with ordinary plots. After all function fills the area between zero and
    plot if the values are above the threshold.
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


def plot_group(data_frame, path_output, dpi):
    # optional import
    import seaborn as sns
    path_output_image = os.path.join(path_output, "summary_statistics.png")

    sns.set_style("whitegrid")
    sns.set_context("talk")
    plt.figure(num=None, figsize=(15, 7), dpi=150)
    plt.ylim(0, 100)
    sns.boxplot(x="Group", y="Stain+ area, %", data=data_frame)
    plt.tight_layout()
    plt.savefig(path_output_image, dpi=dpi)


def main():
    """
    """

    listData = []

    # Pause in seconds between the composite images when --silent(-s) argument is not active
    varPause = 5
    # Initialize the global timer
    startTimeGlobal = timeit.default_timer()
    # Parse the arguments
    args = parse_arguments()

    # load internal resources in json format
    # todo: create the easy selection of stain type from json, add other stains
    # todo: add optimal parameters in json (thresholds and others)
    matrix_json = resources.import_vector()
    parsedJSON = json.loads(matrix_json)
    vectorRawStain = np.array(parsedJSON["vector"])
    str_ch0 = parsedJSON["channel_0"]
    str_ch1 = parsedJSON["channel_1"]
    str_ch2 = parsedJSON["channel_2"]

    pathOutput, pathOutputLog, pathOutputCSV = get_output_paths(args.path)

    check_mkdir_output_path(pathOutput)
    filenames = get_image_filenames(args.path)
    log_and_console(pathOutputLog, "Images for analysis: " + str(len(filenames)), True)
    log_and_console(pathOutputLog, str_ch0 + " threshold = " + str(args.thresh0) +
                    ", " + str_ch1 + " threshold = " + str(args.thresh1) +
                    ", Empty threshold = " + str(args.empty))
    if args.empty>100:
        log_and_console(pathOutputLog, "Empty area filtering is disabled.")
        #log_and_console(pathOutputLog, "It should be adjusted in a case of hollow organ or unavoidable edge defects")

    # Calculate the stain deconvolution matrix
    matrixStains = calc_deconv_matrix(vectorRawStain)

    # Multiprocess implementation
    cores = cpu_count()
    log_and_console(pathOutputLog, "CPU cores used: {}".format(cores))

    # Main cycle where the images are processed and the data is obtained
    pool = Pool(cores)

    wrapper_image_process = partial(image_process, varPause, matrixStains,
                                    args, pathOutput, pathOutputLog, str_ch0, str_ch1, str_ch2)
    for poolResult in pool.imap(wrapper_image_process, filenames):
        listData.append(poolResult)
    pool.close()
    pool.join()

    listFilenames = []
    for filename in filenames:
        listFilenames.append(filename)

    # Creating summary csv after main cycle end
    save_csv(pathOutputCSV, listFilenames, listData)

    # Optional statistical group analysis.
    if args.analyze:
        log_and_console(pathOutputLog,"Group analysis is active")
        group_analyze(filenames, arrayData, pathOutput, args.dpi)
        log_and_console(pathOutputLog,"Statistical data for each group was saved as stats.csv")
        log_and_console(pathOutputLog,"Boxplot with statistics was saved as summary_statistics.png")

    # End of the global timer
    elapsedGlobal = timeit.default_timer() - startTimeGlobal
    if not args.silent:
        averageImageTime = (elapsedGlobal - len(filenames)*varPause)/len(filenames)  # compensate the pause
    else:
        averageImageTime = elapsedGlobal/len(filenames)
    log_and_console(pathOutputLog, "Analysis time: {:.1f} seconds".format(elapsedGlobal))
    log_and_console(pathOutputLog, "Average time per image: {:.1f} seconds".format(averageImageTime))
