This repository contains a lightweight Python tool for detecting structural changes between two Sentinel-1 SAR KMZ files, typically exported from platforms like Copernicus Browser. The tool identifies significant increases in radar backscatter intensity—often indicative of new urban development—by generating a binary difference overlay. The output is a new KMZ file viewable in Google Earth, highlighting detected changes in white.
Key Features:

    Takes two Sentinel-1 KMZ files (older and newer) as input

    Performs threshold-based differencing

    Generates an updated KMZ with detected changes as an overlay

    Fully standalone and requires no use of ESA SNAP toolbox

    Easily extensible: supports custom thresholding and filtering (Gaussian, median)

How to Download Sentinel-1 KMZ Files:

    Visit Copernicus Browser.

    Select Sentinel-1 as the data source and navigate to your area of interest.

    Select the desired acquisition date and click "Visualize".

    Click Download->Analytical and choose Image format = "KMZ/PNG". 

You’ll receive a .kmz file containing the SAR overlay image and its georeferencing, which can be directly used with this script.
