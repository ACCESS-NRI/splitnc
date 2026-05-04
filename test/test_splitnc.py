import pytest
import re

import xarray as xr

from common import runcmd, make_nc
from splitnc import determine_field_vars


@pytest.mark.parametrize(
    "cdl_file,cmd_options,rename_regex,excluded_vars,field_regex,num_nc_files",
    [
        (
            # Test a monthly atmosphere file
            "aiihca.pa-234501_mon.cdl",
            "--shared-vars latitude_longitude --rename-regex {rename_regex}",
            r"(?P<newname>.+)_\d+",
            None,
            "fld_.+",
            217,
        ),
        (
            # Test a daily atmosphere file
            "aiihca.pe-234501_dai.cdl",
            "--shared-vars latitude_longitude --rename-regex {rename_regex}",
            r"(?P<newname>.+)_\d+",
            None,
            "fld_.+",
            36,
        ),
        (
            # Test a monthly ice file
            "iceh-1monthly-mean_2345-01.cdl",
            "--shared-vars uarea,tmask,tarea --excluded-vars VGRDb,VGRDi,VGRDs",
            None,
            ["VGRDb", "VGRDi", "VGRDs"],
            "(ai|dv|si).+",
            53,
        ),
        (
            # Test a daily ice file (use a regex for exluded-vars here)
            "iceh-1daily-mean_2345-01.cdl",
            "--shared-vars uarea,tmask,tarea --excluded-vars VGRD.",
            None,
            ["VGRD."],
            "(ai|dv|si).+",
            25,
        ),
        (
            # Test a monthly atmosphere file with a regex for shared-vars
            # Previously when shared-var regex were resolved after field-var, this failed
            "aiihca.pa-234501_mon.cdl",
            "--shared-vars latitude_lon.+ --rename-regex {rename_regex}",
            r"(?P<newname>.+)_\d+",
            None,
            "fld_.+",
            217,
        ),
        (
            # Test a monthly atmosphere file with a single field with coords that need renaming
            # Previously when the renaming would miss the cell_methods & coordinates
            "aiihca.pa-234501_mon.cdl",
            "--field-vars fld_s03i257 --shared-vars latitude_longitude --rename-regex {rename_regex}",
            r"(?P<newname>.+)_\d+",
            None,
            "fld_.+",
            1,
        ),
        (
            # Test a simple file with time_0 (including a cell_method)
            # Previously when the renaming would miss the cell_methods & coordinates
            "simple_cellmethod_rename.cdl",
            "--shared-vars secondary_field --rename-regex {rename_regex}",
            r"(?P<newname>.+)_\d+",
            None,
            "field",
            1,
        ),
        (
            # Test a daily atmosphere file with a subset of fields
            # Previously fld_s03i236 would trigger a TypeError during renaming
            # due to da.encoding['coordinates']==None
            # Error doesn't trigger with just one field - some detail means
            # coords!=None in that case
            "aiihca.pe-234501_dai.cdl",
            "--field-vars fld_s03i23.* --shared-vars latitude_longitude --rename-regex {rename_regex}",
            r"(?P<newname>.+)_\d+",
            None,
            "fld_s03i23.+",
            6,
        ),
        (
            # Test a simple file with a trailing space in the coords
            "simple_coords_extra_space.cdl",
            "",
            None,
            None,
            "field",
            1,
        ),
    ],
)
@pytest.mark.parametrize("use_cmdline_file", [True, False])
def test_splitnc(tmp_path, cdl_file, cmd_options, rename_regex, excluded_vars,
    field_regex, num_nc_files, use_cmdline_file):
    """
    Test running splitnc from the command line
    """
    # Create a file to test on
    ncfile = make_nc(tmp_path, f"test/data/{cdl_file}")

    output_dir = tmp_path / "single_field"
    
    # Are we using a cmdlinefile?
    if use_cmdline_file:
        cmd_options = cmd_options.format(rename_regex=rename_regex) + \
            f" --output-dir {output_dir} {ncfile}"

        cmdline_file_path = tmp_path / "cmdline_file"
        with open(cmdline_file_path, 'w') as f:
            f.write(cmd_options)

        cmd = f"python splitnc.py --command-line-file {cmdline_file_path}"
    else:
        # Need to mess about with quotes around the regex
        rename_regex = f"'{rename_regex}'"
        cmd_options = cmd_options.format(rename_regex=rename_regex) + \
            f" --output-dir {output_dir} {ncfile}"

        cmd = f"python splitnc.py {cmd_options}"

    # Attempt to split the file
    runcmd(cmd)

    # Check the output files
    output_files = list(output_dir.glob("*.nc"))
    for output_file in output_files:
        ds = xr.open_dataset(
            output_file, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
        )

        # Only one variable in each single-field file should match the field_regex
        count = 0
        for v in ds.variables:
            if re.match(field_regex, v):
                count += 1

        assert count == 1

        # Check none of the variables/coordinates/dims/bounds/cell_methods in
        # the file match the rename regex
        # Also check none of the vars match excluded_vars
        for v in ds.variables:
            if excluded_vars:
                # check excluded_vars don't match v
                for exc_v in excluded_vars:
                    assert not re.match(exc_v, v), \
                    f"{v} - variable should have been excluded"

            if rename_regex:
                # variable name
                assert not re.match(rename_regex, v), \
                    f"{v} - variable hasn't been renamed"

                # dimensions
                assert all([not re.match(rename_regex, d) for d in ds[v].dims]), \
                    f"{v} - dimension hasn't been renamed, {ds[v].dims}"

                # coords from .coords (typically dims + other coords)
                assert all([not re.match(rename_regex, c) for c in ds[v].coords]), \
                    f"{v} - coords hasn't been renamed, {list(ds[v].coords)}"

                # coords from attr (typically just other coords)
                try:
                    coords = ds[v].encoding['coordinates'].split()
                    assert all([not re.match(rename_regex, c) for c in coords]), \
                        f"{v} - coordinate attr hasn't been renamed, {coords}"
                except KeyError:
                    # There will be a KeyError if there are no 'coordinates'
                    pass

                # bounds
                try:
                    bnds = ds[v].attrs['bounds']
                    assert not re.match(rename_regex, bnds), \
                        "{v} - bounds attr hasn't been renamed, {bnds}"
                except KeyError:
                    # There will be a KeyError if there are no 'bounds'
                    pass

                # cell_methods
                try:
                    cell_methods = ds[v].attrs['cell_methods']
                    assert not re.match(rename_regex, cell_methods), \
                        f"{v} - cell_methods hasn't been renamed, {cell_methods}"
                except KeyError:
                    # There will be a KeyError if there are no 'cell_methods'
                    pass

    assert len(output_files) == num_nc_files


@pytest.mark.parametrize(
    "cdl_file,field_regex",
    [
        (
            # Test a simple cdl
            "simple.cdl",
            "field",
        ),
        (
            # Test a simple cdl that has co-dependent fields - i.e. none will be detected
            "simple_circular.cdl",
            "none",
        ),
        (
            # Test a monthly atmosphere file - will also pick up latitude_longitude
            "aiihca.pa-234501_mon.cdl",
            "fld_.+|latitude_longitude",
        ),
        (
            # Test a daily atmosphere file - will also pick up latitude_longitude
            "aiihca.pe-234501_dai.cdl",
            "fld_.+|latitude_longitude",
        ),
        (
            # Test a monthly ice file - will also pick up some extra fields
            "iceh-1monthly-mean_2345-01.cdl",
            "(ai|dv|si|tarea|tmask|uarea|VGRD).*",
        ),
        (
            # Test a daily ice file - will also pick up some extra fields
            "iceh-1daily-mean_2345-01.cdl",
            "(ai|dv|si|tarea|tmask|uarea|VGRD).*",
        ),
    ],
)
def test_determine_field_vars(tmp_path, cdl_file, field_regex):
    """
    Test the functionality for the automatic determinations of field vars
    """
    # Create a file to test on
    ncfile = make_nc(tmp_path, f"test/data/{cdl_file}")

    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(ncfile, decode_times=decoder) as ds:
        field_list = determine_field_vars(ds)

        # Check all the discovered fields match the regex
        assert all([re.match(field_regex, v) for v in field_list])
