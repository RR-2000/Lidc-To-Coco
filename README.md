# LIDC to COCO format

This is a simple script which converts processed numpy image and binary mask files into the COCO annotation format.
The Dicom files where converted to numpy files using the repo: https://github.com/jaeho3690/LIDC-IDRI-Preprocessing

## Required Files
- 2D numpy files with corresponding binary mask numpy files.
- Metadata CSV with tuples for each slice's name and classification category Test/Train/Validation Split.

This 3D numpy to 2D and generation of metadata csv will require existing file structure and metadata format.

## Customization required
- I/O paths need to be changed
  - image_path_annon (line 15)
  - mask_path_annon (line 16)
  - out_path (line 18)
  - Metadata file path (line 22)
- Metadata reading and access format must be changes according to availability. It will depend on the file structure of the data and the csv file.
  - File name Attribute (line 42)
  - Mask name Attribute (line 43)
  - Class Attribute (line 44)
  - Split Attribute (line 45)
- Misc Edits
  - Decide final class for annotation (will be same as annotation from csv file)(line 68-73)
  - Numerical ID for images (line 60-65)
  
