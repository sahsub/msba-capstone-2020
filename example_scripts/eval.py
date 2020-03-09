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
"""

MSBA team: This file takes the predictions and then generates some eval
  metrics. We probably need to talk more about eval metrics. We may
  not get to this. You may see this file referred to as backtesting as
  well but its the same idea (backtesting is eval for time series, doesn't
  really apply to us.)

Combines formatted model predictions with the forecasting features.
The combined data is then used to generate metrics for the evaluation window.
"""

import logging
import os
import sys

import utils

logging.basicConfig(level=logging.DEBUG)


def main():
  """Format model predictions and generate evaluation metrics.

  Combines formatted model predictions with the forecasting features,
  both historical (without predictions) and for the evaluation window.
  The combined data is formatted for use with a Data Studio dashboard
  to plot the forecast and historical demand.
  The combined data is then used to generate metrics for the evaluation
  window specified in the config (based on prediction_start_date and
  horizon_periods).
  It is assumed that the predict script has been run, and the predictions
  are for dates with historical data, not forecasts for dates past the most
  recent data available.
  """

  # Load config shared by all steps of backtest creation.
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

  backtest_params = utils.merge_dicts(
      global_config, query_params['backtest'])

  utils.create_table(
      query_path=os.path.join(
          queries_path, query_files['backtest']),
      query_params=backtest_params,
      destination_project=global_config['project_id'],
      destination_dataset=global_config['forecasting_dataset'],
      destination_table=global_config['backtest_table'],
      partition_field=backtest_params['partition_field'],
  )

  backtest_metrics_params = utils.merge_dicts(
      global_config, query_params['backtest_metrics'])

  utils.create_table(
      query_path=os.path.join(
          queries_path, query_files['backtest_metrics']),
      query_params=backtest_metrics_params,
      destination_project=global_config['project_id'],
      destination_dataset=global_config['forecasting_dataset'],
      destination_table=global_config['backtest_metrics_table'],
      partition_field=None,
  )


if __name__ == '__main__':
  main()