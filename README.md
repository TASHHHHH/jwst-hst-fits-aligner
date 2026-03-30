# JWST/HST FITS Aligner

Python GUI tool for preprocessing multi-filter astronomical datasets (JWST/HST).

## Features
- WCS-based reprojection using FITS headers (Astropy + reproject)
- Strict common-overlap cropping across all aligned frames
- Median / percentile normalization across filters
- Optional PSF matching based on wavelength estimation
- JWST destriping tool for *_rate / *_cal data using segmentation masking
- Batch processing with logging and progress tracking
- GUI interface built with Tkinter

## Motivation
Star-based alignment is unreliable for multi-filter JWST data due to wavelength-dependent PSF variations and inconsistent source appearance.  
This tool uses WCS metadata to align images in celestial coordinates, ensuring consistent alignment across filters.

## Pipeline Overview
1. Load FITS data and headers  
2. Reproject all images to reference WCS  
3. Compute strict common-overlap mask  
4. Crop all images to shared region  
5. Apply normalization (optional)  
6. Apply PSF matching (optional)  
7. Export aligned FITS outputs + logs  

## Tech Stack
- Python
- Astropy (FITS + WCS)
- reproject
- NumPy / SciPy
- photutils
- Tkinter

## Run
```bash
pip install numpy scipy astropy reproject photutils
python aligner.py
