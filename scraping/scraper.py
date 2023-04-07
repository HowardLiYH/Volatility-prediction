import requests
import pandas as pd
import numpy as np
from time import gmtime, strftime
import time
from csv import writer


def main():
    while True:
        r = requests.get("https://api.binance.com/api/v3/depth",
                     params=dict(symbol="BTCBUSD"))
        if r.status_code == 200:
            result = r.json()
            new_row = process(result)

            with open('new_data.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(new_row)

        else:
            print(r.status_code)
        time.sleep(4)


def process(result):
    scrape_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    lastUpdateId = result['lastUpdateId']

    bids = np.array(result['bids'][:20])
    asks = np.array(result['asks'][:20])
    combine = np.concatenate((bids,asks)).reshape((80,))

    final = np.concatenate((np.array([scrape_time, lastUpdateId]),combine))
    return final

main()
