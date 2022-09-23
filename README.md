# smart_receiving

## Project Setup

The code is built and tested on Ubuntu 18.04 and Python 3.7.7

Clone the repository onto the desired location

Download the docker image on the machine, run the following command -
docker pull pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

To spin up the container, run the command in the format -
docker run -it -v <location-of-source-code>:/mnt/host-dir/ -p 8888:8888 -p 5000:5000 <image-id>

For eg: docker run -it -v /mnt/rnd-nlp-cord19/:/mnt/host-dir/ -p 8888:8888 -p 5000:5000 5cc4fb2c1b31

Once you get inside the running container, navigate to the folder where you have mounted your host data folder. Run the command - cd /mnt/host-dir/

# Installing Tesseract OCR as an environment variable

Install Tesseract-OCR and the related developer tools by running the following lines.

sudo apt install tesseract-ocr

sudo apt install libtesseract-dev

Locate the downloaded files by running the commands :

whereis tesseract

/usr/bin/tesseract

## Code Instructions

To run the application on local server, install the project directory having all the codes and dependencies.

Get started after going through the application deliverables on the homepage.

For more information about the application, check the article on smart receiving by clicking on the link provided on the page.

It should be imperative that the vendor selected should correspond to the invoice image being uploaded. 

The invoice image is processed to identify the unique purchase order of the shipment. 

Based on that, the serial numbers and quantities of each individual items are obtained from the invoice.

The final step is to match the OCR generated information with the corresponding shipment information. Minor differences between the extracted data and the original data are corrected and matched. 

A flow diagram for the entire smart receiving workflow is documented below.

![image](https://user-images.githubusercontent.com/109595324/190481417-a674b5b8-bdb3-4cd6-b30c-7e48230cbc4a.png)
