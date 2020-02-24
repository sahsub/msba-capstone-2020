import itertools
import json
import multiprocessing
import os
import time

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery
from google.cloud import language
from google.protobuf.json_format import MessageToJson
import pandas as pd

def load_query_results_to_df(bq_client, query):
    """Loads the results of a query to a dataframe.

        Args:
         bq_client: a BigQuery client.
         query: a BigQuery query.

        Returns:
          A pandas dataframe of the results of the given query.
    """
    job_config = bigquery.QueryJobConfig()
    job_config.use_legacy_sql = False
    query_job = bq_client.query(query, location="US", job_config=job_config)

    df = query_job.to_dataframe()
    df = df.set_index("complaint_id")
    df.index = df.index.astype(str)
    print("Query results loaded into dataframe!")
    return df

def load_checkpoint(checkpoints_path):
    """Loads a list of json dicts from the given path if it exists.
  
      Args:
       checkpoints_path: the directory to use to checkpoint json results.

      Returns:
       A list of dicts, where each dict maps a complaint_id to a response from 
       the NL API.
    """
    if not (os.path.exists(checkpoints_path) and os.listdir(checkpoints_path)):
        os.makedirs(checkpoints_path)
        return [{}]

    stored_results = []
    for filename in sorted(os.listdir(checkpoints_path)):
        path = os.path.join(checkpoints_path, filename)
        if os.path.isfile(path):
            with open(path, "r") as f:
                stored_result = json.load(f)
        stored_results.append(stored_result)
    print("[{0}] Loaded {1} reviews from checkpoint!"
               .format(checkpoints_path, sum(map(len, stored_results))))
    return stored_results

def get_sentiment(complaint, shared_results):
    """Calls the NL sentiment API and stores the response in shared dict.

        This function is used by a pool of multiprocessing workers to write NL API
        responses to a dict stored in shared memory.
  
       Args:
         review: a tuple of (feedback_id, document), where feedback id is the
         id for a review, and document is the review text.
         shared_results: a dict in shared memory of reviews keyed on feedback_id.
     """
    nl_client = language.LanguageServiceClient()
    complaint_id, document = complaint
    try:
        result = nl_client.analyze_sentiment(document=document)
    except GoogleAPICallError as e:
        print(e)
        shared_results[complaint_id] = "language error"
        return
    result = json.loads(MessageToJson(result))
    shared_results[complaint_id] = result
    
def get_entities(complaint, shared_results):
    """Calls the NL entities sentiment API and stores the response in shared dict.

    This function is used by a pool of multiprocessing workers to write NL API
    responses to a dict stored in shared memory.

    Args:
        complaint: a tuple of (feedback_id, document), where complaint_id is the
        id for a complaint, and document is the complaint text.
        shared_results: a dict in shared memory of complaints keyed on complaint_id.
    """
    nl_client = language.LanguageServiceClient()
    complaint_id, document = complaint
    try:
        result = nl_client.analyze_entity_sentiment(document=document)
    except GoogleAPICallError as e:
        print(e)
        shared_results[complaint_id] = "language error"
        return
    result = json.loads(MessageToJson(result))
    shared_results[complaint_id] = result

def call_nl_api(df, stored_results, checkpoints_path, np_api_fn,
                batch_size=2500, max_result_size=50000):
    """Uses the NL API to extract entities and sentiments from reviews in a dataframe.
  
      We ensure that only one batch is called every minute in case we risk hitting the
      API's rate cap. stored_results is used to ensure the API is called only on reviews
      that haven't been seen yet. API responses are stored in the given checkpoints_path
      in shards to avoid writing very large files to disk, where new data is written to
      the most recently created shard until is exceeds max_result_size and a new shard is
      created.

      Args:
        df: a pandas dataframe of complaint narratives.
        stored_results: a list of dicts of json results keyed by complaint_id.
        checkpoints_path: the directory to use to checkpoint json results.
        np_api_fn: a function that calls the NL API.
        batch_size: the number of reviews to process before saving a checkpoint.
        max_result_size: the maximum number of reviews in an output file.

      Returns:
        A dict of json results returned from the NL API.
      """
    batch_timer = time.time()
    batch = []
    num_workers = 8

    # Call the NL API on each complaint in the dataframe.
    for complaint_id, row in df.iterrows():
        if any([complaint_id in res for res in stored_results]):
            continue

        document = language.types.Document(
                     content=row["consumer_complaint_narrative"],
                     type=language.enums.Document.Type.PLAIN_TEXT)
        batch.append((complaint_id, document))
        if len(batch) < batch_size:
            continue
        #import pdb
        #pdb.set_trace()
        try:
            pool= multiprocessing.Pool()
            manager = multiprocessing.Manager()
            batch_results = manager.dict()
            pool.starmap(np_api_fn, zip(batch, itertools.repeat(batch_results)))
            pool.close()
            pool.join()
        except Exception as e:
            print("I am in exception")
            print(e)
        #import pdb
        #pdb.set_trace()
        if len(stored_results[-1]) + len(batch_results) > max_result_size:
            stored_results.append({})
        stored_results[-1].update(batch_results)

        # Write batch to checkpoint, process 1 batch per minute at most.
        delta = time.time() - batch_timer
        filename = "{:05d}.json".format(len(stored_results))
        with open(os.path.join(checkpoints_path, filename), "w+") as f:
            json.dump(stored_results[-1], f)
        total_results = sum(map(len, stored_results))
        print("[{0}] count: {1}".format(checkpoints_path, total_results))
        print("Processed {0} rows in {1} seconds."
                .format(len(batch_results), delta))

        delta = time.time() - batch_timer
        if delta < 60:
            time.sleep(60 - delta)
        batch_timer = time.time()
        batch = []

    # Write remaining complaints in batch one last time.
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    batch_results = manager.dict()
        
    pool.starmap(np_api_fn, zip(batch, itertools.repeat(batch_results)))
    pool.close()
    pool.join()
    stored_results[-1].update(batch_results)

    filename = "{:05d}.json".format(len(stored_results))
    with open(os.path.join(checkpoints_path, filename), "w+") as f:
        json.dump(stored_results[-1], f)
    total_results = sum(map(len, stored_results))
    print("[{0}] final count: {1}".format(checkpoints_path, total_results))
    return stored_results

def add_sentiment_data(df, stored_results):
    sentiment_scores = []
    sentiment_magnitudes = []
    index = []
    for stored_result in stored_results:
        for complaint_id, result in stored_result.items():
            index.append(complaint_id)
            if result == "language error":
                sentiment_scores.append(0)
                sentiment_magnitudes.append(0)
                continue
            score = magnitude = 0
            if "score" in result["documentSentiment"]:
                score = result["documentSentiment"]["score"]
            if "magnitude" in result["documentSentiment"]:
                magnitude = result["documentSentiment"]["magnitude"]
            sentiment_scores.append(score)
            sentiment_magnitudes.append(magnitude)
    df["sentiment_score"] = pd.Series(sentiment_scores, index)
    df["sentiment_magnitude"] = pd.Series(sentiment_magnitudes, index)

    return df

def add_entity_data(df, stored_results):
    """Adds entity data to the given dataframe.

    Args:
      df: a pandas dataframe.
      stored_results: a list of dicts of json results returned from the NL API.

    Returns:
      A pandas dataframe of reviews with additional columns for entities.
    """
    entities_names = []
    entities_types = []
    entities_sent_scrs = []
    entities_sent_mags = []
    index = []
    for stored_result in stored_results:
        for complaint_id, result in stored_result.items():
            index.append(complaint_id)
            if result == "language error":
                entities_names.append([])
                entities_types.append([])
                entities_sent_scrs.append([])
                entities_sent_mags.append([])
                continue
            entity_names = []
            entity_types = []
            entity_sent_scrs = []
            entity_sent_mags = []
            if "entities" in result:
                for entity in result["entities"]:
                    score = magnitude = 0
                    if "score" in entity["sentiment"]:
                        score = entity["sentiment"]["score"]
                    if "magnitude" in entity["sentiment"]:
                        magnitude = entity["sentiment"]["magnitude"]
                    entity_names.append(entity["name"])
                    entity_types.append(entity["type"])
                    entity_sent_scrs.append(score)
                    entity_sent_mags.append(magnitude)
            entities_names.append(entity_names)
            entities_types.append(entity_types)
            entities_sent_scrs.append(entity_sent_scrs)
            entities_sent_mags.append(entity_sent_mags)
    df["entities"] = pd.Series(entities_names, index)
    df["entity_types"] = pd.Series(entities_types, index)
    df["entity_sentiment_scores"] = pd.Series(entities_sent_scrs, index)
    df["entity_sentiment_magnitudes"] = pd.Series(entities_sent_mags, index)
    return df