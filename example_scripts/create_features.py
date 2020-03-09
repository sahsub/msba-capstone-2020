# Copyright 2020 Google Inc. All Rights Reserved.
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
"""Runs SQL queries to reates Bigquery tables for training and prediction.

MSBA Team: This file should call the NLP API logic (ok if in separate file),
  write it to a table, then query the source data and combine it with the
  NLP API information while cleaning and write to a new table for training/eval.
  Maybe also split that table into training and evaluation and write to a
  separate table? All the code in here can be deleted.

"""

import logging
import os
import sys

import utils

logging.basicConfig(level=logging.DEBUG)


def main():
  """Runs queries to create training and prediction tables from clean data."""

  # Load config shared by all steps of feature creation.
  config_path = utils.parse_arguments(sys.argv).config_path
  config = utils.read_config(config_path)
  # Project-wide config.
  global_config = config['global']
  # Path to SQL files.
  queries_path = config['file_paths']['queries']
  # SQL files for different pipeline steps.
  query_files = config['query_files']
  # Parameters unique to individual pipeline steps.
  query_params = config['query_params']

  # Create the dataset to hold data for the pipeline run.
  utils.create_dataset(
      destination_project=global_config['project_id'],
      destination_dataset=global_config['forecasting_dataset'],
  )

  # Query to assemble weekly sales from clean data.
  weekly_sales_params = utils.merge_dicts(global_config,
                                          query_params['weekly_sales'])

  # Format the list of departments into a single query parameter.
  # Ex. a list ['210', '132'] is formatted as "'210', '132'"
  weekly_sales_params['departments'] = ', '.join(
      ["'{}'".format(s) for s in weekly_sales_params['departments']])

  utils.create_table(
      query_path=os.path.join(queries_path, query_files['weekly_sales']),
      query_params=weekly_sales_params,
      destination_project=global_config['project_id'],
      destination_dataset=global_config['forecasting_dataset'],
      destination_table=global_config['weekly_sales_table'],
      partition_field=weekly_sales_params['partition_field'],
  )

  forecasting_features_params = utils.merge_dicts(
      global_config, query_params['forecasting_features'])

  utils.create_table(
      query_path=os.path.join(
          queries_path, query_files['forecasting_features']),
      query_params=forecasting_features_params,
      destination_project=global_config['project_id'],
      destination_dataset=global_config['forecasting_dataset'],
      destination_table=global_config['forecasting_features_table'],
      partition_field=forecasting_features_params['partition_field'],
  )

  forecasting_features_train_params = utils.merge_dicts(
      global_config, query_params['forecasting_features_train'])

  utils.create_table(
      query_path=os.path.join(
          queries_path, query_files['forecasting_features_split']),
      query_params=forecasting_features_train_params,
      destination_project=global_config['project_id'],
      destination_dataset=global_config['forecasting_dataset'],
      destination_table=global_config['forecasting_features_train_table'],
      partition_field=forecasting_features_train_params['partition_field'],
  )

  forecasting_features_predict_params = utils.merge_dicts(
      global_config, query_params['forecasting_features_predict'])

  utils.create_table(
      query_path=os.path.join(
          queries_path, query_files['forecasting_features_split']),
      query_params=forecasting_features_predict_params,
      destination_project=global_config['project_id'],
      destination_dataset=global_config['forecasting_dataset'],
      destination_table=global_config['forecasting_features_predict_table'],
      partition_field=forecasting_features_predict_params['partition_field'],
  )

  # To add more steps to the pipeline, you just need to copy some variation of
  # the two commands above -- merge the global parameters and the query_params
  # for the new query, and then run utils.create_table with the correct
  # input.


if __name__ == '__main__':
  main()
