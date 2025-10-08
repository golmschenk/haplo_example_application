from pathlib import Path

from torch.optim import AdamW

from haplo.internal.dataset.split import split_dataset_into_count_datasets
from haplo.internal.dataset.xarray_zarr import XarrayBasedDataset
from haplo.internal.distributed import distributed_logging
from haplo.internal.losses import SumDifferenceSquaredOverMedianExpectedSquaredMetric, PlusOneChiSquaredStatisticMetric
from haplo.internal.train_hyperparameter_configuration import TrainHyperparameterConfiguration
from haplo.internal.train_logging_configuration import TrainLoggingConfiguration
from haplo.internal.train_session import train_session
from haplo.internal.train_system_configuration import TrainSystemConfiguration
from haplo.internal.transforms.affine_normalize import default_output_affine_transform, default_input_affine_transform
from haplo.models import Cura


@distributed_logging
def example_train_session():
    dataset_path = Path('data/2k_parameters_and_phase_amplitudes.zarr.zip')
    full_dataset = XarrayBasedDataset.new(zarr_path=dataset_path)
    test_dataset, validation_dataset, train_dataset, _ = split_dataset_into_count_datasets(
        full_dataset, [200, 200, 1_600])
    model = Cura.new(input_transformation=default_input_affine_transform,
                     output_transformation=default_output_affine_transform)
    loss_function = SumDifferenceSquaredOverMedianExpectedSquaredMetric()
    metric_functions = [PlusOneChiSquaredStatisticMetric(),
                        SumDifferenceSquaredOverMedianExpectedSquaredMetric()]
    hyperparameter_configuration = TrainHyperparameterConfiguration.new(cycles=10)
    system_configuration = TrainSystemConfiguration.new()
    optimizer = AdamW(params=model.parameters(), lr=hyperparameter_configuration.learning_rate,
                      weight_decay=hyperparameter_configuration.weight_decay,
                      eps=hyperparameter_configuration.optimizer_epsilon)
    run_comments = f'Example run.'  # Whatever you want to log in a string.
    additional_log_dictionary = {
        'model_name': type(model).__name__, 'train_dataset_size': len(train_dataset), 'run_comments': run_comments
    }
    logging_configuration = TrainLoggingConfiguration.new(
        wandb_project='example', wandb_entity='ramjet', additional_log_dictionary=additional_log_dictionary)
    train_session(train_dataset=train_dataset, validation_dataset=validation_dataset, model=model,
                  loss_function=loss_function, metric_functions=metric_functions, optimizer=optimizer,
                  hyperparameter_configuration=hyperparameter_configuration, system_configuration=system_configuration,
                  logging_configuration=logging_configuration)


if __name__ == '__main__':
    example_train_session()
