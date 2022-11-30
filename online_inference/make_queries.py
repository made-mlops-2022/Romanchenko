import click
import pandas as pd


@click.command()
@click.option(
    '--address',
    default='127.0.0.1:80',
    help='Address of the service.'
)
@click.option(
    '-n',
    default='1',
    help='Number of requests '
)
def shoot(address, n):
    all_queries = pd.read_csv('sample_queries.csv')
    for i in range(n):
        q = all_queries.iloc[i]
        q[""]
