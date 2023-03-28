import requests
import pandas as pd
import numpy as np
from time import gmtime, strftime
import time
from google.cloud import bigquery
import os


PROJECT = os.environ.get('GCP_PROJECT')
DATASET = os.environ.get('BQ_DATASET')
CRAWL_INTERVAL = int(os.environ.get('CRAWL_INTERVAL'))
TABLE = 'all'

table = f"{PROJECT}.{DATASET}.{TABLE}"


scrape_time = []

def main():
    client = bigquery.Client()
    write_mode = "WRITE_APPEND" # "WRITE_TRUNCATE" or "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    while True:
        r = requests.get("https://api.binance.com/api/v3/depth",
                     params=dict(symbol="BTCBUSD"))
        if r.status_code == 200:

            result = r.json()

            df = process(result)

            client.load_table_from_dataframe(df, table, job_config=job_config)

        else:
            print(r.status_code)
        time.sleep(CRAWL_INTERVAL)


def process(result):
    t = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    lastUpdateId = result['lastUpdateId']
    bids = np.array(result['bids'][:20])
    asks = np.array(result['asks'][:20])
    combine = np.concatenate((bids,asks)).reshape((1,80))

    columns = []
    for i in range(1,21):
        columns.append(f"bp_{i}")
        columns.append(f"bz_{i}")

    for i in range(1,21):
        columns.append(f"ap_{i}")
        columns.append(f"az_{i}")

    final = pd.DataFrame(combine, columns=columns)
    final['scrape_time'] = t
    final['lastUpdateId'] = lastUpdateId
    return final

main()
