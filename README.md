# splitnc
This script splits multi-field netCDF files into single-field files.
It is designed to work on ESM1.6's atmosphere and ice files.

## Automatic Field Identification
By default `splitnc` will attempt to identify the fields for a multi-field netCDF files by looking for variables that no other variables depend on.
A variable that no others depend on is likely to be a field.
E.g. many variables depend on `time`, but none depend on `sea_surface_temperature`.

Alternatively the fields to separate to individual files can be specified as a comma separated list with the `--field-vars` command line option.
`--field-vars` interprets each item as regex, e.g. one could use `--field-vars fld_.+` to match all variable names that start with the string `fld_`.

## "Ancillary" Variables

Some variables with no dependents should not be separated into individual files, these variables must be manually identified with the `--shared-vars` command line option.
These variables will then be present in every output file.
Regex is also supported for this option.

If there are ancillary fields that should only be present in only some of the output field files then multiple invocations of `splitnc` using `--field-vars` and `--shared-vars` will be required.

Example of these variables are the `latitude_longitude` found in atmosphere files or the `uarea`, `tmask`, `tarea`, `VGRDb`, `VGRDi`, `VGRDs` variables from ice files.

## Config File

The `-c`/`--command-line-file` option can be used to supply a filepath to a file that contains command line options.
If this option is used, all other options supplied on the command line will be ignored.
Newline characters in the file will be treated as whitespace, i.e. newlines can be used as well as spaces to separate command line arguments.

For example to replicate this command line,
```
splitnc --verbose --overwrite --output-dir /output/directory --shared-vars latitude_longitude --rename-regex "(?P<newname>.+)_\d+" /input/directory/*.nc
```
the following file could be used;
```
--verbose
--overwrite
--output-dir /output/directory
--shared-vars latitude_longitude
--rename-regex "(?P<newname>.+)_\d+"
/input/directory/*.nc
```

## Command Line Options

```quote
usage: splitnc [-h] [--field-vars FIELD_VAR1,FIELD_VAR2,...] [--shared-vars SHARED_VAR1,SHARED_VAR2,...]
               [--output-name-pattern OUTPUT_NAME_PATTERN] [--rename-regex REGEX] [--output-dir OUTPUT_DIR] [--overwrite] [-v]
               [-c COMMAND_LINE_FILE]
               [filepaths ...]

Splits a multi-field netCDF file into separate one-field files

positional arguments:
  filepaths             One or more filepaths to process

options:
  -h, --help            show this help message and exit
  --field-vars FIELD_VAR1,FIELD_VAR2,...
                        Specify the names of the field variables to split into separate files - dimensions, bounds, and
                        coordinates of these fields will be included in each file. Disables automatic field variable
                        identification. Regex patterns can be used here.
  --shared-vars SHARED_VAR1,SHARED_VAR2,...
                        Specify the names of variables that should be shared across files that cannot be automatically
                        identified, as a comma separated list. Regex patterns can be used here.
  --excluded-vars EXCLUDED_VAR1,EXCLUDED_VAR2,...
                        Specify the names of variables that should be excluded from files. This option can be used with
                        automatic identification of field variables. Regex patterns can be used here.
  --rename-regex REGEX  Look for duplicated coordinate names that match the given regex and rename them to the first
                        "newname" capture group in the regex. E.g. "(?P<newname>.*)_\d+" will match "time_0" and rename
                        it to "time".
  --use-esm1p6-filenames
                        Use the ESM1.6 filename pattern for the output files:
                        access-esm1p6.{component}.{dimensions}.{field}.{freq}.{time_cell_method}.{datestamp}.nc
                        splitnc will attempt to deduce all the components of the filename. If this option is not given
                        {field}_{original_filename} will be used.
  --fix-cell-methods    Correct cell_methods by adding 'time: point' to cell_methods for variables that have 'time' but
                        not 'time_bnds' and no other 'time' cell_methods.
  --file-freq FILE_FREQ
                        Specify the frequency of the files (not the data), e.g. if each file contains a month of data
                        then the file-frequency is '1mon'. Used to determine the resolution of the timestamp for ESM1.6
                        filenames. Follows the ACCESS frequency vocabulary (e.g. '1yr', '1mon', '1day', '1hr'), any
                        unrecognised frequency will use the full timestamp. Defaults to '1yr'.
  --output-dir OUTPUT_DIR
                        Output directory for the processed files. If not given output files will be placed in the same
                        directory as the original file.
  --overwrite           Overwrite existing files
  --dont-update-history
                        Disable automatic update of history attribute
  -v, --verbose
  -c COMMAND_LINE_FILE, --command-line-file COMMAND_LINE_FILE
                        A file containing a list of command-line arguments. Newlines in this file will be ignored. If
                        supplied all other command line arguments will be ignored.
```

## Example Usage

`splitnc` just needs the `xarray` and `netCDF4` python modules.
On Gadi use load any module with `xarray`, such as `conda/analysis3`.
Alternatively create a new python environment and install `xarray` and `netCDF4`.

### Atmosphere
To use this script for split multi-field atmosphere files from ACCESS-ESM1.6:
```bash
splitnc --shared-vars latitude_longitude  --rename-regex "(?P<newname>.+)_\\d+" $INPUT_DIR/*.nc
```

`splitnc` will automatically determine which variables are fields by looking at which variables depend on other variables.
Variables with nothing depending on them are deemed to be fields.
Alternatively one could use `--field-vars fld_.+` to match the variable names in these files.

The `--rename-regex` option with the supplied regex will rename variables like
`time_0` or `pseudo_level_0` are renamed to `time` or `pseudo_level`.

The `--shared-vars` option will ensure that the variable `latitude_longitude` is
included in all files even though none of the field variable depend on it.

### Ice
To use this script for split multi-field ice files from ACCESS-ESM1.6:
```bash
splitnc --shared-vars uarea,tmask,tarea --excluded-vars VGRD. $INPUT_DIR/*.nc
```

In comparison to the atmosphere files, ice files have different shared-vars and there are no duplicated variables that require renaming.
The variables `VGRDb`, `VGRDi`, and `VGRDs` are not required and can thus be excluded from the output.
