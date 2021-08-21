# Lidc to Coco format

This is a simple Script which converts processed numpy image and binary mask files into the COCO annotation format.

## Required Files
- 2D numpy files with corresponding binary mask numpy files
- Metadata CSV with tuples for each slice's name and classification category Test/Train/Validation Split 

This 3D numpy to 2D and generation of metadata csv will require existing file structure and metadata format

## Customization required
- I/O paths need to be changed
  - image_path_annon
  - mask_path_annon
  - out_path
- Metadata reading and access format must be changes according to availability
  - File name Attribute
  - Mask name Attribute
  - Class Attribute
  - Split Attribute
- Misc Edits
  - Decide final class for annotation
  - Numerical ID for images
  
