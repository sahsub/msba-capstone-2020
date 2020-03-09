-- Copyright 2020 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--    http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
-- =============================================================================

/*
 * MSBA Team: Here's an example of a query. Note the use of Python3 {format}
 *   substitutions. The queries are read in by Python and values are substituted
 *   into the {format} blocks from the yaml file.
 */
SELECT *
FROM `{project_id}.{source_dataset}.{source_table}`
