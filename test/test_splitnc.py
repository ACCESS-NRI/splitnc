import os
from pathlib import Path
import pytest
import re

import xarray as xr

from common import runcmd, make_nc
from splitnc import determine_field_vars, build_filename


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
            218,
        ),
        (
            # Test a monthly atmosphere file with esm1.6 filenames
            "aiihca.pa-234501_mon.cdl",
            "--shared-vars latitude_longitude --rename-regex {rename_regex} --use-esm1p6-filenames",
            r"(?P<newname>.+)_\d+",
            None,
            "fld_.+",
            218,
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
            # Test a daily atmosphere file with esm1.6 filenames
            "aiihca.pe-234501_dai.cdl",
            "--shared-vars latitude_longitude --rename-regex {rename_regex} --use-esm1p6-filenames",
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
            # Test a monthly ice file with esm1.6 filenames
            "iceh-1monthly-mean_2345-01.cdl",
            "--shared-vars uarea,tmask,tarea --excluded-vars VGRDb,VGRDi,VGRDs --use-esm1p6-filenames",
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
            # Test a daily ice file (use a regex for exluded-vars here) with esm1.6 filenames
            "iceh-1daily-mean_2345-01.cdl",
            "--shared-vars uarea,tmask,tarea --excluded-vars VGRD. --use-esm1p6-filenames",
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
            218,
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
    
    cmd_options += f" --output-dir {output_dir} {ncfile}"

    # Are we using a cmdlinefile?
    if use_cmdline_file:
        cmd_options = cmd_options.format(rename_regex=rename_regex)

        cmdline_file_path = tmp_path / "cmdline_file"
        with open(cmdline_file_path, 'w') as f:
            f.write(cmd_options)

        cmd = f"splitnc --command-line-file {cmdline_file_path}"
    else:
        # Need to mess about with quotes around the regex
        rename_regex = f"'{rename_regex}'"
        cmd_options = cmd_options.format(rename_regex=rename_regex)

        cmd = f"splitnc {cmd_options}"

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

    os.remove(ncfile)


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

    os.remove(ncfile)


@pytest.mark.parametrize("use_esm1p6", [True, False])
@pytest.mark.parametrize(
    "cdl_file,field,output_freq,expected_filename",
    [
        (
            # Test a monthly atmos 2D field
            "aiihca.pa-234501_mon.cdl",
            "fld_s00i023",
            "1yr",
            "access-esm1p6.um7p3.2d.fld_s00i023.1mon.mean.2345.nc",
        ),
        (
            # Test a monthly atmos 2D field but monthly output
            "aiihca.pa-234501_mon.cdl",
            "fld_s00i023",
            "1mon",
            "access-esm1p6.um7p3.2d.fld_s00i023.1mon.mean.2345-01.nc",
        ),
        (
            # Test a monthly atmos 3D field
            "aiihca.pa-234501_mon.cdl",
            "fld_s00i407",
            "1yr",
            "access-esm1p6.um7p3.3d.fld_s00i407.1mon.mean.2345.nc",
        ),
        (
            # Test a daily atmos 3D field
            "aiihca.pe-234501_dai.cdl",
            "fld_s30i207",
            "1yr",
            "access-esm1p6.um7p3.3d.fld_s30i207.1day.mean.2345.nc",
        ),
        (
            # Test a daily ice 3D field
            "iceh-1daily-mean_2345-01.cdl",
            "siitdconc",
            "1yr",
            "access-esm1p6.cice5.3d.siitdconc.1day.mean.2345.nc",
        ),
        (
            # Test a daily ice fx field
            "iceh-1daily-mean_2345-01.cdl",
            "tarea",
            "1yr",
            "access-esm1p6.cice5.2d.tarea.fx.nc",
        ),
        (
            # Test a 2-hourly ice 2D field
            "iceh-2hourly-mean_0272.cdl",
            "siconc",
            "1yr",
            "access-esm1p6.cice5.2d.siconc.2hr.mean.0272.nc",
        ),
        (
            # Test a 2-daily ice 2D field
            "iceh-2daily-mean_0272.cdl",
            "siconc",
            "1yr",
            "access-esm1p6.cice5.2d.siconc.2day.mean.0272.nc",
        ),
        (
            # Test a 2-monthly ice 2D field
            "iceh-2monthly-mean_0272.cdl",
            "siconc",
            "1yr",
            "access-esm1p6.cice5.2d.siconc.2mon.mean.0272.nc",
        ),
        (
            # Test an hourly ice 2D field
            "iceh-1hourly-mean_0272.cdl",
            "siconc",
            "1yr",
            "access-esm1p6.cice5.2d.siconc.1hr.mean.0272.nc",
        ),
        (
            # Test an hourly instantaneous ice 2D field
            # This variable has time_rep = instantaneous
            # This field was manually added to the .cdl file
            "iceh-1hourly-mean_0272.cdl",
            "siconc2",
            "1yr",
            "access-esm1p6.cice5.2d.siconc2.1hr.snap.0272.nc",
        ),
        (
            # Test an hourly instantaneous ice 2D field
            # This variable has time: point in the cell_methods
            # This field was manually added to the .cdl file
            "iceh-1hourly-mean_0272.cdl",
            "siconc3",
            "1yr",
            "access-esm1p6.cice5.2d.siconc3.1hr.snap.0272.nc",
        ),
        (
            # Test an timestep/hourly ice 2D field
            # The frequency of timestep files is not defined, so it will fail
            # to build an ESM1.6 filename
            # The time bounds for this file are also incorrect but we expect a
            # failure anyway so it doesn't matter
            "iceh-1-mean_0272.cdl",
            "siconc",
            "1yr",
            ValueError("Unable to deduce frequency"),
        ),
        (
            "iceh-1yearly-mean_0272.cdl",
            "siconc",
            "1yr",
            "access-esm1p6.cice5.2d.siconc.1yr.mean.0272.nc",
        ),
        (
            # Test a yearly ice 2D field with the year manually tweaked to be 0001
            "iceh-1yearly-mean_0001.cdl",
            "siconc",
            "1yr",
            "access-esm1p6.cice5.2d.siconc.1yr.mean.0001.nc",
        ),
        (
            # Test an hourly 2d atmos field
            "aiihca.pc-010101.cdl",
            "fld_s05i216",
            "1yr",
            "access-esm1p6.um7p3.2d.fld_s05i216.1hr.mean.0101.nc",
        ),
        (
            # Test a 3-hourly 2d atmos field
            "aiihca.pi-010101_3hr.cdl",
            "fld_s00i409",
            "1yr",
            "access-esm1p6.um7p3.2d.fld_s00i409.3hr.snap.0101.nc",
        ),
        (
            # Test a 6-hourly 2d atmos field
            "aiihca.pj-010101_6hr.cdl",
            "fld_s03i245",
            "1yr",
            "access-esm1p6.um7p3.2d.fld_s03i245.6hr.mean.0101.nc",
        ),
    ]
)
def test_build_filenames(tmp_path, use_esm1p6, cdl_file, field, output_freq, expected_filename):
    # Create a file to test on
    ncfile = make_nc(tmp_path, f"test/data/{cdl_file}")

    def _build_filename():
        decoder = xr.coders.CFDatetimeCoder(time_unit='us')
        with xr.open_dataset(ncfile, decode_times=decoder) as ds:
            actual_filename = build_filename(
                ds,
                field,
                Path(cdl_file.replace('.cdl', '.nc')),
                esm1p6_filename=use_esm1p6,
                file_freq=output_freq,
            )

        return actual_filename

    if use_esm1p6 and isinstance(expected_filename, Exception):
        with pytest.raises(type(expected_filename), match=str(expected_filename)):
            _ = _build_filename()
    else:
        actual_filename = _build_filename()

        if not use_esm1p6:
            # If we're not using the ESM1.6 filepattern we expect field_file.nc
            expected_filename = f"{field}_{Path(cdl_file.replace('.cdl', '.nc'))}"

        assert actual_filename == expected_filename

        os.remove(ncfile)
