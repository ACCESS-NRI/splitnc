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


def process_file(
    filepath,
    field_vars=None,
    shared_vars=None,
    excluded_vars=None,
    rename_regex=None,
    output_dir=None,
    overwrite=False,
    update_history=True,
):
    logging.debug(f"Processing {filepath}")
    filepath = Path(filepath)

    # Use cftime to suppress warnings
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(filepath, decode_times=decoder) as ds:
        # Resolve any regex in the excluded_vars list
        if excluded_vars:
            excluded_vars = match_regex_list(excluded_vars, ds.variables)
        else:
            excluded_vars = []
        logging.debug(f"List of defined excluded variables is: {excluded_vars}")

        # Resolve any regex in the shared_vars list
        if shared_vars:
            shared_vars = match_regex_list(shared_vars, ds.variables)

            # shared_vars should not be in excluded vars
            shared_vars = [v for v in shared_vars if v not in excluded_vars]
        else:
            shared_vars = []
        logging.debug(f"List of defined shared variables is: {shared_vars}")

        # Determine the field vars
        if field_vars is None or len(field_vars) == 0:
            logging.debug("Automatically determining field variables")

            field_vars = determine_field_vars(ds)
        else:
            # There may be regex to process
            field_vars = match_regex_list(field_vars, ds.variables)

        # Shared and excluded vars shouldn't be field_vars
        logging.debug("Removing shared variables from list of field variables")
        field_vars = [v for v in field_vars if v not in shared_vars and v not in excluded_vars]
        logging.debug(f"List of field vars is: {field_vars}")

        # Build the mapping dict for renaming, e.g. {"time_0: "time"}
        if rename_regex:
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
            if update_history:
                new_history = build_history()
                logging.debug(f"Updating history attribute with: {new_history}")
                update_history_attr(ds_v, new_history)

            if output_dir:
                output_dir = Path(output_dir)
            else:
                output_dir = filepath.parent

            output_filename = output_dir / f"{v}_{filepath.name}"
            logging.debug(f"Output filepath is {output_filename}")

            if not overwrite and output_filename.exists():
                logging.error(f"Output file already exists - {output_filename}")
                logging.error("Use --overwrite to overwrite existing files")

                raise FileExistsError(f"{output_filename} already exists")

            logging.debug("Creating parent directory and writing to output file")
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            ds_v.to_netcdf(output_filename)


#### Main
def arg_parse(cmdline_args=None):
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
        nargs="*",
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
        "--output-dir",
        help="Output directory for the processed files. If not given output "
        "files will be placed in the same directory as the original file.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--dont-update-history",
        action="store_true",
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
    args.filepaths = [
        filepath for glob_list in args.filepaths for filepath in glob_list
    ]

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

    for f in args.filepaths:
        process_file(
            f,
            field_vars=args.field_vars,
            shared_vars=args.shared_vars,
            excluded_vars=args.excluded_vars,
            rename_regex=args.rename_regex,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            update_history=not args.dont_update_history,
        )


if __name__ == "__main__":
    main()
