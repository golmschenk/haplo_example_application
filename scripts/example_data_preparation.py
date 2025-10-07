from pathlib import Path

from haplo.internal.data_conversion import constantinos_kalapotharakos_format_file_to_xarray_zarr_zip


def example_data_preparation():
    constantinos_kalapotharakos_format_file_to_xarray_zarr_zip(
        Path('data/mcmc_vac_all_2k.dat'), Path('data/2k_parameters_and_phase_amplitudes.zarr.zip'))


if __name__ == '__main__':
    example_data_preparation()
