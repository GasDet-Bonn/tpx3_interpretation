# Interpreation of tpx3-daq data

The purpose of this python script is to interpret raw Timepix3 data recorded
with [tpx3-daq](https://github.com/GasDet-Bonn/tpx3-daq). The script expects
as input an HDF5 file with the raw data that was created by the readout
software. Furthermore a path to a new HDF5 file is expected. The interpreted
data and the chip configuration will be written to the new HDF5 file.

## Requirements
The following python packages are needed to run the script:
```
numpy
tables
```
Additionally the package [basil](https://github.com/SiLab-Bonn/basil) needs
to be installed.

## Usage
The script can be used with
```
python3 tpx3_interpretation.py <path_to_raw_data.h5> <path_for_new_output.h5>
```

## Output
The script crates a new HDF5 file with the following content:

    - interpreted
        - run_0
            - configuration
                - dacs
                - generalConfig
                - links
                - mask_config
                - thr_matrix
            - hit_data

The HDF5 output can be used as input for
[TimepixAnalysis](https://github.com/Vindaar/TimepixAnalysis) for using first
the `raw_data_manipulation` and then the `reconstruction`.
