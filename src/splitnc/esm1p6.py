import logging
import re


def _build_model():
    # Model is always access-esm1p6
    return "access-esm1p6"


def _build_component(ds):
    # Component: either CICE5 or UM7.3
    source = ds.attrs["source"]
    if "Los Alamos Sea Ice Model (CICE) Version 5" in source:
        return "cice5"
    elif "Data from Met Office Unified Model" in source and \
        ds.attrs['um_version'] == "7.3":
        return "um7p3"
    else:
        raise ValueError(f"Unknown source, {source}")


def _build_dimensions(ds, field_name):
    # Dimensions: Don't count time when seeing if field is 2d or 3d
    ndims = len([d for d in ds[field_name].dims if d!='time'])
    if ndims == 2:
        return "2d"
    elif ndims == 3:
        return "3d"
    else:
        raise ValueError(f"Unexpected number for dimensions, {ndims}")


def _build_frequency(ds, field_name, input_filepath):
    # Frequency: use fx if no time dim
    if 'time' not in ds[field_name].dims:
        return "fx"

    # Attempt to parse from expected filenames
    filename = input_filepath.name

    # Define the expected ice filenames
    # e.g. iceh-2hourly-mean_0272.nc, iceh-1yearly-mean_0272.nc
    ice_regex = r"iceh-(?P<num>\d+)(?P<unit>yearly|monthly|daily|hourly)-"
    ice_unit_mapping = {
        "yearly": "yr",
        "monthly": "mon",
        "daily": "day",
        "hourly": "hr"
    }

    if match:=re.match(ice_regex, filename):
        # Extract the frequency number and units for ice files
        return f"{match['num']}{ice_unit_mapping[match['unit']]}"
    elif "_mon.nc" in filename:
        # Match the monthly pattern for atmosphere files
        return "1mon"
    elif "_dai.nc" in filename:
        # Match the daily pattern for atmosphere files
        return "1day"
    elif match:=re.match(r".+_(\d+hr).nc", filename):
        # Get the frequency from the atmosphere regex match for Xhr
        return match[1]
    elif "aiihca.pc" in filename:
        # Match another pattern for hourly atmosphere files
        return "1hr"

    # No sub-hourly frequency data expected
    raise ValueError("Unable to deduce frequency from filename")


def _build_cell_method(ds, field_name):
    attrs = ds[field_name].attrs

    try:
        if attrs['time_rep'] == "instantaneous":
            # ice files sometimes have time_rep = instantaneous but not
            # cell_methods = time: point
            return ".snap"
    except KeyError:
        # Continue if 'time_rep' not in attrs
        pass

    # Time cell_method: Should be able to deduce from the cell_method
    cell_method_regx = r"time: (\w+)"
    try:    
        if m:= re.search(cell_method_regx, attrs["cell_methods"]):
            method = m[1]
            if method == "point":
                method = "snap"

            # Since this element is optional add the . here
            return "." + method
    except KeyError:
        # Continue if 'cell_methods' not in attrs
        pass

    # If there's time but no time_bnds and no time cell_method then assume snap
    # This case is intended to catch instantaneous atmospheric fields from um2nc
    if "time" in ds and "bounds" not in ds["time"].attrs:
        return ".snap"

    # Otherwise omit this element from the filename
    return ""


def _build_datestamp(ds, field_name, file_freq):
    if 'time' not in ds[field_name].dims:
        # No datetime for fixed files
        return ""

    # Truncate average time val by output file frequency
    # datetimes do not correctly zero-pad so need to use %4Y
    if re.match(r'\d+(yr|dec)', file_freq):
        fmt = '%4Y'
    elif re.match(r'\d+mon', file_freq):
        fmt = '%4Y-%m'
    elif re.match(r'\d+day', file_freq):
        fmt = '%4Y-%m-%d'
    else:
        fmt = '%4Y-%m-%dT%H:%M:%S'

    # Get the appropriately truncated datetime for the average time
    try:
        # Try the time bounds
        time_arr = ds[ds['time'].attrs["bounds"]]
        logging.debug("Using time bounds to calculate filename timestamp")
    except KeyError:
        # If there are no time bounds just use time
        logging.debug("Unable to find time bounds, using time to calculate filename timestamp")
        time_arr = ds['time']

    # Calculate the middle point
    first, last = time_arr.min(), time_arr.max()
    datestamp_dt = (first + (last - first) / 2).compute().dt
        # Need to .compute when using open_mfdataset

    return "." + datestamp_dt.strftime(fmt).data.flatten()[0]


def build_esm1p6_filename(ds, field_name, input_filepath, esm1p6_filename=False, file_freq="1yr"):
    template = "{model}.{component}.{dimensions}.{field}.{freq}{time_cell_method}{datestamp}.nc"

    # Model is always access-esm1p6
    try:
        d = {
            "model": _build_model(),
            "component": _build_component(ds),
            "dimensions": _build_dimensions(ds, field_name),
            "field": field_name,
            "freq": _build_frequency(ds, field_name, input_filepath),
            "time_cell_method": _build_cell_method(ds, field_name),
            "datestamp": _build_datestamp(ds, field_name, file_freq),
        }
    except ValueError as e:
        # Reraise the exception with some extra information
        e.args = (*e.args, f"While building output filename for field {field_name} and {input_filepath}")
        raise

    return template.format(**d)
