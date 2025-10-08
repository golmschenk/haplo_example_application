from pathlib import Path

import numpy as np
import torch
from bokeh.io import show
from bokeh.plotting import figure

from haplo.internal.dataset.split import split_dataset_into_count_datasets
from haplo.internal.dataset.xarray_zarr import XarrayBasedDataset
from haplo.internal.export import WrappedModel
from haplo.internal.transforms.affine_normalize import default_input_affine_transform, default_output_affine_transform
from haplo.models import Cura


def example_infer_session():
    dataset_path = Path('data/2k_parameters_and_phase_amplitudes.zarr.zip')
    full_dataset = XarrayBasedDataset.new(zarr_path=dataset_path)
    test_dataset, validation_dataset, train_dataset, _ = split_dataset_into_count_datasets(
        full_dataset, [200, 200, 1_600])

    model = Cura.new(input_transformation=default_input_affine_transform,
                     output_transformation=default_output_affine_transform)
    model = WrappedModel(model)  # The DDP module requires an extra wrapping. This emulates that.
    saved_model_path = Path('sessions/your/path/to/model.pt')
    model.load_state_dict(torch.load(str(saved_model_path), map_location=torch.device('cpu')))
    model.eval()

    test_parameters0, test_phase_amplitudes0 = test_dataset[0]
    input_array = np.expand_dims(test_parameters0, axis=0)
    with torch.no_grad():
        input_tensor = torch.tensor(input_array)
        output_tensor = model(input_tensor)
        output_array = output_tensor.numpy()
    predicted_test_phase_amplitudes0 = np.squeeze(output_array, axis=0)

    comparison_figure = figure(x_axis_label='phase', y_axis_label='amplitude')
    comparison_figure.line(x=list(range(64)), y=test_phase_amplitudes0, line_color='mediumblue')
    comparison_figure.line(x=list(range(64)), y=predicted_test_phase_amplitudes0, line_color='firebrick')
    show(comparison_figure)


if __name__ == '__main__':
    example_infer_session()
