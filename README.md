# MorphoStain
Python soft for morphometric stain analysis. Software counts the stained area with determined stain (DAB-chromagen for example) using the typical immunohistochemistry protocols. After the analysis user can measure the difference of proteins content in tested samples.

### Contents:
1. [Application](#Application)
2. [Requirements](#Requirements)
3. [Installation](#Installation)
4. [Basic principles](#Basic-principles)
5. [Image samples requirements](#Image-samples-requirements)
6. [Interface type](#Interface-type)
7. [Composite image examples](#Composite-image-examples)
8. [Summary statistics](#Summary-statistics-image-example)
9. [Log](#Log-example)
10. [CSV output](#CSV-output-example)
11. [Statistical data output](#Statistical-data-output-example)
12. [Command line arguments](#Command-line-arguments)
13. [Typical options usage](#Typical-options-usage)
14. [Authorship](#Authorship)
15. [Acknowledgements](#Acknowledgements)



### Application:
Quantitative analysis of extracellular matrix proteins in IHC-analysis, designed for scientists in biotech sphere. Could be also used for general morphometric analysis.

### Requirements:
Python 2.7 or Python 3.4-3.5

Python libraries: numpy, scipy, skimage, matplotlib, pandas

Optional (for group analysis): seaborn

### Installation:
Install **pip** using your system package manager. For example in Debian/Ubuntu:

```
sudo apt-get install python3-pip
```

Clone this repository

In the root folder of repository clone perform:

```
sudo pip3 install .
```

Uninstall:
```
sudo pip3 uninstall morphostain
```

### Interface type:
No GUI, command line interface only.

### Basic principles:
Script uses the **color deconvolution method**. It was well described by [G. Landini](http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html). Python port from Skimage package of his algorythm was used. See also: *Ruifrok AC, Johnston DA. Quantification of histochemical staining by color deconvolution. Anal Quant Cytol Histol 23: 291-299, 2001.*

Color deconvolution is used to separate stains in multi-stained sample. This soft is mainly applied for Hematoxyline + DAB staining. Script uses hardcoded stain matrix or custom one in JSON format. You should determine your own for better result using ImageJ and hyperlink above. Determined custom matrix should replace the default one. For additional information see the comments in code.

After stain separation, script determines the stain-positive area using the default or user-defined threshold. The empty areas are excluded from the final relative area measurement as the sample could contain free space, which would affect the result accuracy.

Script creates the result folder inside the --path. Statistics, log and composite images for each sample are saved there.

### Image samples requirements
1. Image samples' **white balance should be normalized**! It is important to get the right colors of stains before separation. I could suggest free software like [Rawtherapee](http://rawtherapee.com/)
2. Images should be acquired using the **same exposure values**
3. Threshold should be the same at the whole image sequence if you want to compare them
4. It would be better to use the manual mode in microscope camera to be sure, that your images were taken with the same parameters.
5. Don't change light intensity in microscope during the sequence acquiring.
6. Correct file naming should be used if group analysis is active. Everything before _ symbol will be recognized as a group name.


### Examples
You can find test images in this repository.

#### Composite image examples
Script will render this type of image for **each of your samples**. User should control the result to be sure that the threshold values are right
![Composite image example 1](https://github.com/meklon/DAB_analyzer/blob/master/test%20images/result%20example/Native_Pan_05_analysis.png "Composite image example")

![Composite image example 2](https://github.com/meklon/DAB_analyzer/blob/master/test%20images/result%20example/Alex_Pan_08_analysis.png "Composite image example")

#### Summary statistics image example
![Stat image example](https://github.com/meklon/DAB_analyzer/blob/master/test%20images/result%20example/summary_statistics.png "Stat image example")
How to read box plot:

![](http://i1.wp.com/flowingdata.com/wp-content/uploads/2008/02/box-plot-explained.gif?w=1090)
#### Log example
```
Images for analysis: 62
Stain threshold = 40, Empty threshold = 101
Empty area filtering is disabled.
It should be adjusted in a case of hollow organ or unavoidable edge defects
CPU cores used: 2
Image saved: /home/meklon/temp/sample_native/result/Col1_02_analysis.png
Image saved: /home/meklon/temp/sample_native/result/Col1_01_analysis.png
Image saved: /home/meklon/temp/sample_native/result/Col4_02_analysis.png
Image saved: /home/meklon/temp/sample_native/result/Col4_03_analysis.png
Group analysis is active
Statistical data for each group was saved as stats.csv
Boxplot with statistics was saved as summary_statistics.png
Analysis time: 44.3 seconds
Average time per image: 0.7 seconds
```
#### CSV output example
Filename | Stain+ area, %
------------ | -------------
Alex_Pan_06.jpg|61.55
Native_Pan_05.jpg|14.23
Native_Trop_02.jpg|10.83

#### Statistical data output example
Group|mean|std|median|amin|amax
------------ | -------------| -------------| -------------| -------------| -------------|
Col1|38.906666666666666|11.818569075823012|37.16|24.58|61.12
Col4|30.514444444444443|9.177221953171763|30.12|16.62|45.66
Fibr|38.287499999999994|7.836421832881198|34.875|30.41|53.51
Lam|34.327777777777776|8.20530130125911|33.02|21.88|46.8
Pan|10.21375|7.495407998997023|7.29|2.92|21.97
Trop|13.702000000000002|3.9725329171421317|14.235|7.22|20.34
VEGF|6.644444444444444|5.6577117969880515|4.84|0.96|16.7

### Command line arguments
Place all the sample images (8-bit) inside the separate folder. Subdirectories are excluded from analysis. Use the following options:

*-p, --path* (obligate) - path to the target directory with samples

*-t0, --thresh0* (optional) - Global threshold for stain-positive area of channel_0 stain.

*-t1, --thresh1* (optional) - Global threshold for stain-positive area of channel_2 stain.

*-e, --empty* (optional) - threshold for **empty area** separation. If empty the default value would be used (threshEmptyDefault = 101). It is disabled for default and should be used only in a case of hollow organs and unavoidable edge defects.

*-s, --silent* (optional) - if True, the real-time composite image visualisation would be supressed. The output will be just saved in the result folder.

*-a, --analyze* (optional) - Add group analysis after the indvidual image processing. The groups are created using the filename. Everything before _ symbol will be recognized as a group name. Example: **Native_10.jpg, Native_11.jpg** will be counted as a single group **Native**.

*-m, --matrix* (optional) - Your matrix in a JSON formatted file. Could be used for alternative stain vectors. Not for regular use yet. Test in progress.

####Typical options usage
````
morphostain -p /home/meklon/Data/sample/test/ -t0 35 -t1 -e 89 -s -a
````

### JSON structure
Stain vectors and predefined values like typical thresholds are stored in JSON format. By default dab.json is loaded. You can also use your own one with --matrix option.
Histogram shift values are used to normalize image histogram.
Typical structure:
```json
{
  "channel_0":"Hematoxylin",
  "channel_1":"DAB-chromogen",
  "channel_2":"Supplementary channel",
  "vector": [[0.66504073, 0.61772484, 0.41968665],
            [0.4100872, 0.5751321, 0.70785],
            [0.6241389, 0.53632, 0.56816506]],
  "thresh_0":30,
  "thresh_1":40,
  "hist_shift_0":5,
  "hist_shift_1":18,
  "hist_shift_2":0
}

```

### Authorship
Gumenyuk Ivan, Kuban state medical university, Russia.

### Acknowledgements
Special thanks to my teammates from our lab, Eugene Dvoretsky (@radioxoma), Alexey Lavrenuke (@direvius) and everyone, who helped me with this work.