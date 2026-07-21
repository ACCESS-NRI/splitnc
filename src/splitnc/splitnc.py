import argparse
from collections import Counter
from datetime import datetime, timezone
from glob import glob
import logging
from pathlib import Path
from platform import python_version
import re
import sys

import xarray as xr

from splitnc.esm1p6 import build_esm1p6_filename


def determine_field_vars(ds):
    """
    Attempt to determine which variables in the xarray dataset are fields

    If a variable is not depended on by any other variables it is likely to be a
    field. E.g.
        F(x, y, z), x, y, z
            F depends on x, y, and z so x, y, and z are not fields
            Nothing depends on F, so F is likely to be a field

    Need to check dimensions, bounds, and coordinates
    """
    reference_counts = Counter()

    for varname in ds.variables:
        # Any dims that are not variables will be ignored
        reference_counts.update(ds[varname].dims)

        try:
            reference_counts.update(ds[varname].encoding["coordinates"].split())
        except KeyError:
            pass

        try:
            reference_counts.update([ds[varname].attrs["bounds"]])
        except KeyError:
            pass

    return sorted(
        [varname for varname in ds.variables if reference_counts[varname] == 0]
    )


def get_dependent_vars(ds, varname, curr_vars=None):
    """
    Get a list of variables that the given variable depends on.

    Check dimensions, bounds, and coordinates

    Recurse on each NEW dependent to get other dependents.

    By only recursing on new dependents infinite recursion in the case of
    circular dependencies is avoided.
    """
    logging.debug(f"Determining dependent variables for {varname}")
    if curr_vars is None:
        curr_vars = set()

    # Get any dims that are also variables
    new_vars = {d for d in ds[varname].dims if d in ds.variables}

    # Get any coords
    if (
        "coordinates" in ds[varname].encoding
        and ds[varname].encoding["coordinates"] is not None
    ):
        new_vars.update(ds[varname].encoding["coordinates"].split())

    # Add bounds if the variable has them
    if "bounds" in ds[varname].attrs:
        bounds = ds[varname].attrs["bounds"]
        new_vars.update([bounds])

    # Get the set of vars that are actually new (to avoid infinite recursion)
    diff_vars = new_vars.difference(curr_vars)

    all_vars = curr_vars | new_vars

    # Recurse on each new var
    additional_vars = set()
    for new_v in diff_vars:
        additional_vars |= get_dependent_vars(ds, new_v, all_vars)

    return diff_vars | additional_vars


def get_vars_in_order(ds, varname):
    """
    Get the variables in order

    - Start with the field for this dataset,
    - Followed by the dimensions of the field
      - each dim followed by their bounds if they exist
    - Finish with anything remaining in alphabetical order
    """
    # Order the variables
    vars_to_order = list(ds.variables)

    # Start with the field
    vars_in_order = [varname]
    vars_to_order.remove(varname)

    # Then the field's dimension and their bnds in order
    for dim_name in ds[varname].dims:
        if dim_name not in vars_to_order:
            continue

        vars_in_order.append(dim_name)
        vars_to_order.remove(dim_name)
        if "bounds" in ds[dim_name].attrs:
            dim_bnd_name = ds[dim_name].attrs["bounds"]
            if dim_bnd_name in vars_to_order:
                vars_in_order.append(dim_bnd_name)
                vars_to_order.remove(dim_bnd_name)

    # Then the remaining variables in alphabetical order
    vars_in_order += sorted(vars_to_order)

    return vars_in_order


def rename_variable(ds, oldname, newname):
    """
    Rename a variable, xarray handles most of the rename.

    If the variable has a bounds variable, also rename the matching portion of
    the bound's name. I.e. latitude -> lat therefore latitude_bnds -> lat_bnds
    """
    logging.debug(f"Renaming {oldname} to {newname}")
    ds_new = ds.rename({oldname: newname})

    for v in ds.variables:
        # Update cell_methods
        try:
            old_cell_methods = ds_new[v].attrs['cell_methods']
            if old_cell_methods and oldname in old_cell_methods:
                new_cell_methods = old_cell_methods.replace(oldname, newname)
                logging.debug(f"Renaming {oldname} to {newname} in {v}'s cell_methods - {old_cell_methods} to {new_cell_methods}")
                ds_new[v].attrs['cell_methods'] = new_cell_methods
        except KeyError:
            # Do nothing if there's no cell_methods
            pass

        # Update coordinates
        try:
            old_coords = ds_new[v].encoding['coordinates']
            if old_coords and oldname in old_coords:
                new_coords = old_coords.replace(oldname, newname)
                logging.debug(f"Renaming {oldname} to {newname} in {v}'s coordinates - {old_coords} to {new_coords}")
                ds_new[v].encoding['coordinates'] = new_coords
        except KeyError:
            # Do nothing if there's no coords
            pass

    # Update bounds
    try:
        old_bnd_name = ds_new[newname].attrs["bounds"]
        new_bnd_name = old_bnd_name.replace(oldname, newname)

        logging.debug(f"Renaming {old_bnd_name} to {new_bnd_name}")
        ds_new = rename_variable(ds_new, old_bnd_name, new_bnd_name)

        # Update the attr on the original variable
        logging.debug(f'Updating "bounds" attr on {newname} to {new_bnd_name}')
        ds_new[newname].attrs["bounds"] = new_bnd_name
    except KeyError:
        # This variable doesn't have bounds
        pass

    return ds_new


def match_regex_list(regex_list, string_list):
    """
    Return strings in the given list that match the supplied regex
    """
    compiled_regex = [re.compile(regex) for regex in regex_list]
    return [s for s in string_list if any(r.fullmatch(s) for r in compiled_regex)]


def build_rename_dict(ds, rename_regex):
    """
    Use the supplied regex to build a dictionary of {"oldname": "newname"} to
    pass to xarray's rename.

    "newname" should be supplied as a named capture group in the regex.

    E.g. to rename "time_0", "time_1" or "height_0" to "time" or "height", one
    could use the regex "(?P<newname>.+)_\\d+".
    """
    logging.debug("Building rename dict")
    rename_dict = {}
    for coord in ds.coords:
        m = re.fullmatch(rename_regex, str(coord))

        if m:
            try:
                newname = m["newname"]
            except IndexError as e:
                logging.error(
                    f"{coord} matched regex for renaming, {rename_regex}, "
                    'but no "newname" capture group found'
                )
                raise e

            logging.debug(f"{coord} will be renamed to {newname}")

            rename_dict[coord] = newname

    return rename_dict


def build_history():
    time_stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    python_exe = f"python{python_version()}"

    # The list of files given on the commandline is not needed in the history
    args = " ".join(sys.argv)
  
    return f"{time_stamp} : splitnc (https://github.com/ACCESS-NRI/esm1.6-scripts) : {python_exe} {args}"


def update_history_attr(ds, new_history):
    if "history" in ds.attrs:
        old_history = ds.attrs["history"] + "\n"
    else:
        old_history = ""

    ds.attrs["history"] = old_history + new_history


def fix_cell_methods(ds, varname):
    """
    Fix missing cell_methods for instantaneous variables from um2nc.

    If variable has 'time' but no time 'bounds' and there are no other 'time'
    cell_methods then add 'time: point' to the cell_methods.
    """
    if "time" in ds and "bounds" not in ds["time"].attrs:
        try:
            cell_methods = ds[varname].attrs["cell_methods"]
        except KeyError:
            cell_methods = ""

        if "time" not in cell_methods:
            new_cell_methods = f"{cell_methods} time: point".strip()
            logging.debug(f"Updating cell_methods for {varname} to {new_cell_methods}")
            ds[varname].attrs["cell_methods"] = new_cell_methods


def build_filename(ds, field_name, input_filepath, esm1p6_filename=False, file_freq="1yr"):
    """
    Build the filename used for the output.

    If esm1p6_filename=False then <field_name>_<orginal_file_name> will be used.

    Otherwise a filename that follows the ESM1.6 naming scheme will be used:
    {model}.{component}.{dimensions}.{field}.{freq}.{time_cell_method}.{datestamp}.nc
    More info here: https://access-om3-configs.access-hive.org.au/configurations/Ocean_diagnostics/
    Elements of this schema will be deduced from the Dataset, the original filename,
    and the given output file frequency.
    """
    if esm1p6_filename:
        return build_esm1p6_filename(ds, field_name, input_filepath,
            esm1p6_filename=esm1p6_filename, file_freq=file_freq)
    else:
        return f"{field_name}_{input_filepath.name}"


def group_filepaths(filepaths, group_regex):
    r"""
    Group together files from the list of filepaths that match the given regex
    with only the portion in the capture group "wild" varying.

    E.g. if files follow the patterns
    - aiihca.pa-YYYYMM_mon.nc and
    - aiihca.pe-YYYYMM_dai.nc
    use "aiihca\.p[ae]-\d{4}(?P<wild>\d{2})_(mon|dai)\.nc" to group together
    months for each year and freq. Grouped filepaths will be returned as a list

    Any filepath that doesn't match the regex will be returned alone, i.e. in a
    group of length 1.

    Returns a list of lists of filepath strings
    """
    grouped_filepaths = []
    while len(filepaths) > 0:
        f = filepaths[0]
        if m:=re.search(group_regex, f):
            # We need to know which indices the "wild" group has in the filepath
            wild_span = m.span("wild")
            
            # Replace the wild match in the orginal string with a wild regex
            # Use double {{ }} to escape them in f-strings
            group_regx = re.compile(m.string[:wild_span[0]] + f".{{{len(m['wild'])}}}" + m.string[wild_span[1]:])

            # Get the filepaths that match the regex and remove them from the filepaths list
            group_list = [fp for fp in filepaths if group_regx.search(fp)]
            filepaths = [fp for fp in filepaths if not group_regx.search(fp)]
        else:
            # If the regex doesn't match the group regex treat it as a solo group
            group_list = [filepaths.pop(0)]

        grouped_filepaths.append(group_list)

    return grouped_filepaths


def process_files(**kwargs):
    # Prepare the filepath list
    filepaths_list = kwargs.pop("filepaths")
    if input_group_regex:=kwargs['input_group_regex']:
        logging.debug(f"Grouping filepaths according to regex: {input_group_regex}")

        # Group files together according to the input_file_date_regex
        filepaths_list = group_filepaths(filepaths_list, input_group_regex)
    else:
        # Treat every filepath as a size 1 group
        filepaths_list = [[f] for f in filepaths_list]
    
    logging.debug("Filepaths groups as follows:\n" + "\n".join(
        [f"{i}: {filepaths}" for i, filepaths in enumerate(filepaths_list)]
    ))

    # Process each filepath group
    for filepaths in filepaths_list:
        process_filegroup(filepaths, **kwargs)


def process_filegroup(filepaths, **kwargs):
    # Define default kwargs and update them with kwargs
    kwargs = {
        "excluded_vars": [],
        "shared_vars": [],
        "field_vars": None,
        "rename_regex": None,
        "update_history": True,
        "fix_cell_methods": False,
        "output_dir": False,
        "use_esm1p6_filenames": False,
        "file_freq": "1yr",
        "overwrite": False,
    } | kwargs

    logging.debug(f"Processing {filepaths}")

    filepaths = [Path(f) for f in filepaths]
    
    # xarray drops .encoding when using open_mfdataset with more than one file
    # So save the encodings when loading and reapply
    encoding_map = {}
    def save_encoding(ds):
        for v in ds.variables:
            enc = ds[v].encoding
            
            # Remove source (i.e. filename) from encoding, those will never match
            del enc['source']

            if v in encoding_map and encoding_map[v] != enc:
                raise ValueError(f"Encodings for {v} doesn't match across all files: {enc}")

            encoding_map[v] = enc

        return ds

    # Use cftime to suppress warnings
    decoder = xr.coders.CFDatetimeCoder(time_unit='us')
    with xr.open_mfdataset(filepaths, decode_times=decoder, combine="nested", 
        compat="no_conflicts", join="outer", preprocess=save_encoding) as ds:
        # Reapply the saved encodings if they're missing
        for v in ds.variables:
            if not ds[v].encoding:
                ds[v].encoding = encoding_map[v]

        # Resolve any regex in the excluded_vars list
        if excluded_vars:=kwargs["excluded_vars"]:
            excluded_vars = match_regex_list(excluded_vars, ds.variables)
        logging.debug(f"List of defined excluded variables is: {excluded_vars}")

        # Resolve any regex in the shared_vars list
        if shared_vars:=kwargs["shared_vars"]:
            shared_vars = match_regex_list(shared_vars, ds.variables)

            # shared_vars should not be in excluded vars
            shared_vars = [v for v in shared_vars if v not in excluded_vars]
        logging.debug(f"List of defined shared variables is: {shared_vars}")

        # Determine the field vars
        if field_vars:=kwargs["field_vars"]:
            # There may be regex to process
            field_vars = match_regex_list(field_vars, ds.variables)
        else:
            logging.debug("Automatically determining field variables")
            field_vars = determine_field_vars(ds)

        # Shared and excluded vars shouldn't be field_vars
        logging.debug("Removing shared variables from list of field variables")
        field_vars = [v for v in field_vars if v not in shared_vars and v not in excluded_vars]
        logging.debug(f"List of field vars is: {field_vars}")

        # Build the mapping dict for renaming, e.g. {"time_0: "time"}
        if rename_regex:=kwargs["rename_regex"]:
            rename_dict = build_rename_dict(ds, rename_regex)
        else:
            rename_dict = {}
        logging.debug(f"Rename dict is {rename_dict}")

        for v in field_vars:
            # Get the list of vars to keep for this field
            logging.debug(f"Determining dependent variables for field variable {v}")
            dependent_vars = get_dependent_vars(ds, v)
            full_var_list = [v] + list(dependent_vars) + shared_vars

            # Drop any vars not in the list
            drop_vars_list = [v for v in ds.variables if v not in full_var_list]
            ds_v = ds.drop_vars(drop_vars_list)

            # Rename anything in the rename dict
            if rename_dict:
                for old_name, new_name in rename_dict.items():
                    if (
                        old_name in ds_v.variables
                        or old_name in ds_v.dims
                        or old_name in ds_v.coords
                    ):
                        ds_v = rename_variable(ds_v, old_name, new_name)

            # Coordinates shouldn't have _FillValues
            for coord in list(ds_v.coords):
                if coord in ds_v.variables:
                    logging.debug(f'Setting "_FillValue" to None for {coord}')
                    ds_v[coord].encoding["_FillValue"] = None

            # Bounds shouldn't have coordinates or _FillValues
            bnds_set = {
                ds_v[bnd_v].attrs["bounds"]
                for bnd_v in ds_v.variables
                if "bounds" in ds_v[bnd_v].attrs
            }
            logging.debug(f"Bounds variables are {bnds_set}")
            for bnd in bnds_set:
                logging.debug(
                    f'Setting "coordinates" and "_FillValue" to None for {bnd}'
                )
                ds_v[bnd].encoding["coordinates"] = None
                ds_v[bnd].encoding["_FillValue"] = None

            # Order the variables
            vars_in_order = get_vars_in_order(ds_v, v)
            logging.debug(f"Ordering variable as {vars_in_order}")
            ds_v = ds_v[vars_in_order]

            # Update the history attribute
            if kwargs["update_history"]:
                new_history = build_history()
                logging.debug(f"Updating history attribute with: {new_history}")
                update_history_attr(ds_v, new_history)

            # Fix cell_methods
            if kwargs["fix_cell_methods"]:
                fix_cell_methods(ds_v, v)

            # Output path construction assumes the first path can be used
            if output_dir:=kwargs["output_dir"]:
                output_dir = Path(output_dir)
            else:
                output_dir = filepaths[0].parent

            # Build the output filepath
            filename = build_filename(
                ds=ds_v,
                field_name=v,
                input_filepath=filepaths[0],
                esm1p6_filename=kwargs["use_esm1p6_filenames"],
                file_freq=kwargs["file_freq"],
            )
            output_filepath = output_dir / filename
            logging.debug(f"Output filepath is {output_filepath}")

            # Write to file
            if not kwargs["overwrite"] and output_filepath.exists():
                logging.error(f"Output file already exists - {output_filepath}")
                logging.error("Use --overwrite to overwrite existing files")

                raise FileExistsError(f"{output_filepath} already exists")

            logging.debug("Creating parent directory and writing to output file")
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            ds_v.to_netcdf(output_filepath)


#### Main
def arg_parse(cmdline_args=None):
    # If -c/--command-line-file is being used then all other args are ignored
    # This affects which are "required" (or nargs for filepaths)
    args = sys.argv if cmdline_args is None else cmdline_args
    cmd_file_arg_present = "-c" in args or "--command-line-file" in args

    parser = argparse.ArgumentParser(
        prog="splitnc",
        description="Splits a multi-field netCDF file into separate one-field files",
    )

    # Create a custom type for comma separated strings as lists
    def comma_separated_string_type(s):
        return s.split(",")

    # Open the named file and parse it as a command line split it around the
    # whitespaces (including newlines)
    def command_line_file(filepath):
        with open(filepath, "r") as f:
            file_str = f.read()

        return file_str.split()

    # Filepath wildcards won't be expanded if supplied via a command line file
    # I.e. *.nc won't be expanded by the shell to [file1.nc, file2.nc]
    def globbable_string_list(string_list):
        return glob(string_list)

    # Let filepaths be optional (i.e. nargs=* instead of +) so that it isn't
    # required and --cmd-line-file can be used on it's own
    parser.add_argument(
        "filepaths",
        nargs="*" if cmd_file_arg_present else "+",
        default=[],
        type=globbable_string_list,
        help="One or more filepaths to process",
    )
    parser.add_argument(
        "--field-vars",
        type=comma_separated_string_type,
        default=[],
        metavar="FIELD_VAR1,FIELD_VAR2,...",
        help="Specify the names of the field variables to split into separate "
        "files - dimensions, bounds, and coordinates of these fields will "
        "be included in each file. Disables automatic field variable "
        "identification. Regex patterns can be used here.",
    )
    parser.add_argument(
        "--shared-vars",
        type=comma_separated_string_type,
        default=[],
        metavar="SHARED_VAR1,SHARED_VAR2,...",
        help="Specify the names of variables that should be shared across "
        "files that cannot be automatically identified, as a comma "
        "separated list. Regex patterns can be used here.",
    )
    parser.add_argument(
        "--excluded-vars",
        type=comma_separated_string_type,
        default=[],
        metavar="EXCLUDED_VAR1,EXCLUDED_VAR2,...",
        help="Specify the names of variables that should be excluded from "
        "files. This option can be used with automatic identification of field "
        "variables. Regex patterns can be used here.",
    )
    parser.add_argument(
        "--rename-regex",
        metavar="REGEX",
        help="Look for duplicated coordinate names that match the given regex "
        'and rename them to the first "newname" capture group in the '
        'regex. E.g. "(?P<newname>.*)_\\d+" will match "time_0" and '
        'rename it to "time".',
    )
    parser.add_argument(
        "--use-esm1p6-filenames",
        action="store_true",
        help="Use the ESM1.6 filename pattern for the output files: "
        "access-esm1p6.{component}.{dimensions}.{field}.{freq}.{time_cell_method}.{datestamp}.nc"
        " splitnc will attempt to deduce all the components of the filename. "
        "If this option is not given {field}_{original_filename} will be used."
    )
    parser.add_argument(
        "--fix-cell-methods",
        action="store_true",
        help="Correct cell_methods by adding 'time: point' to cell_methods "
        "for variables that have 'time' but not 'time_bnds' and no other "
        "'time' cell_methods."
    )
    parser.add_argument(
        "--file-freq",
        default="1yr",
        help="Specify the frequency of the files (not the data), e.g. if each "
        "file contains a month of data then the file-frequency is '1mon'. Used "
        "to determine the resolution of the timestamp for ESM1.6 filenames. "
        "Follows the ACCESS frequency vocabulary (e.g. '1yr', '1mon', '1day', "
        "'1hr'), any unrecognised frequency will use the full timestamp. "
        "Defaults to '1yr'."
    )
    parser.add_argument(
        "--input-group-regex",
        help="Specify a regex that will be used to group a subset of the input "
        "into a single set. E.g. group together 12 input monthly files to "
        "a single year of output. Use a named capture group \"wild\" to "
        "specify the portion of the filename that varies. E.g. to group monthly "
        "files with this pattern - \"aiihca.pa-YYYYMM_mon.nc\" - use the regex "
        r"\"aiihca\.pa-\d{4}(?P<wild>\d{2})_mon\.nc\"."
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for the processed files. If not given output "
        "files will be placed in the same directory as the original file.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    # By default update the history attr
    # To avoid passing around a negative store_false and rename this arg
    parser.add_argument(
        "--dont-update-history",
        action="store_false",
        dest="update_history",
        help="Disable automatic update of history attribute"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--command-line-file",
        type=command_line_file,
        help="A file containing a list of command-line arguments. Newlines in "
        "this file will be ignored. If supplied all other command line "
        "arguments will be ignored.",
    )

    args = parser.parse_args(args=cmdline_args)

    # File paths may need flattened since glob was used
    # Sort the list to ensure repeatable behaviour
    args.filepaths = sorted([
        filepath for glob_list in args.filepaths for filepath in glob_list
    ])

    # If the command line yaml was supplied use the contents instead of argv
    if args.command_line_file:
        return arg_parse(args.command_line_file)
    else:
        return args


def setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )


def main():
    args = arg_parse()

    setup_logging(args.verbose)

    logging.debug(f"Command line args are: {args}")

    if len(args.filepaths) == 0:
        logging.error("No files to process.")
        raise ValueError("No files to process.")

    process_files(**vars(args))


if __name__ == "__main__":
    main()
