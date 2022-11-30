import json

import click
import pandas as pd
import requests as r


@click.command()
@click.option(
    '--address',
    default='http://127.0.0.1:80',
    help='Address of the service.'
)
@click.option(
    '-n',
    default='1',
    help='Number of requests '
)
def shoot(address, n):
    print("Shoooot")
    n = int(n)
    all_queries = pd.read_csv('sample_queries.csv').to_dict('records')
    for i in range(n):
        q = all_queries[i % len(all_queries)]
        response = r.post(address + "/predict", data=json.dumps(q))
        print(f'Result status code: {response.status_code}, prediction: {response.json()["prediction"]}')


if __name__ == '__main__':
    shoot()
