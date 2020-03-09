# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Trains a model with AutoML Forecasting using parameters specified in the
config. Assumes that forecasting features have been generated. Training
typically completes in one hour more than that specified in the train
budget, though compute is only charged for the budgeted time.
See https://cloud.google.com/automl-tables/docs/train for details.

MSBA team: this is a training example using Tables Forecasting which works
differently. And this is out of date. This file should create the AutoML
dataset and then kick the training.
"""

import logging
import sys

from google.cloud import automl_v1beta1 as automl
from google.oauth2 import service_account

import utils

logging.basicConfig(level=logging.DEBUG)


def main():
  """Executes training for a model using the AutoML Forecasting service.

  Uses parameters specified in the configuration file, including definitions
  of feature (ex. data type as categorical or numeric) as well as the
  optimization objective. See the configuration file for more details.
  """
  config_path = utils.parse_arguments(sys.argv).config_path
  config = utils.read_config(config_path)

  # Authenticate using AutoML service account.
  credentials = service_account.Credentials.from_service_account_file(
      config['file_paths']['automl_service_account_key'])

  # Defining subconfigs explicitly for readability.
  global_config = config['global']
  model_config = config['model']

  client = automl.TablesClient(
      project=global_config['project_id'],
      region=global_config['automl_compute_region'],
      credentials=credentials,
  )

  bigquery_uri_train_table = 'bq://{}.{}.{}'.format(
      global_config['project_id'],
      global_config['forecasting_dataset'],
      global_config['forecasting_features_train_table'],
  )

  # Set the dataset type to forecasting explicitly, otherwise it assumes that
  # it is a regression problem and displays incorrectly in the UI.
  dataset = client.create_dataset(
      global_config['dataset_display_name'],
      metadata={'tables_dataset_type': 'FORECASTING'},
  )

  # Import operation is a Long Running Operation, .result() performs a
  # synchronous wait for the import to complete before progressing.
  import_data_operation = client.import_data(
      dataset=dataset,
      bigquery_input_uri=bigquery_uri_train_table,
  )
  import_data_operation.result()

  # Update the data type, nullability, and forecasting type
  # (ex. time independent metadata). Assumes fields are defined for every col.
  for column_spec_display_name, column in model_config['columns'].items():
    client.update_column_spec(
        dataset=dataset,
        column_spec_display_name=column_spec_display_name,
        type_code=column['type_code'],
        nullable=column['nullable'],
        forecasting_type=column['forecasting_type'],
    )

  # Time column to index time series.
  client.set_time_column(
      dataset=dataset,
      column_spec_display_name=model_config['time_column'],
  )

  # Target column to predict, historical values will be used for prediction.
  client.set_target_column(
      dataset=dataset,
      column_spec_display_name=model_config['target_column'],
  )

  # Column to define a manual split of the dataset into "TRAIN",
  # "VALIDATE", and "TEST".
  client.set_test_train_column(
      dataset=dataset,
      column_spec_display_name=model_config['split_column'],
  )

  # Weights the training loss for each row (not time series).
  client.set_weight_column(
      dataset=dataset,
      column_spec_display_name=model_config['weight_column'],
  )


  # Tunes and trains model, expect a ~ 1 hour overhead in addition to the time
  # allowed by the training budget. Stops tuning early if no further
  # improvements are made.
  create_model_response = client.create_model(
      model_display_name=global_config['model_display_name'],
      dataset=dataset,
      prediction_type='FORECASTING',
      granularity_unit=global_config['granularity_unit'],
      horizon_periods=global_config['horizon_periods'],
      train_budget_milli_node_hours=(
          1000 * model_config['train_budget_hours']),
      exclude_column_spec_names=model_config['exclude_columns'],
  )
  create_model_response.result()

if __name__ == '__main__':
  main()