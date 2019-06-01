# coding:utf-8
import os
import psycopg2
import numpy as np
import pandas as pd

def gradecd(x):
    if x == 'A':
        return 0
    if x == 'B':
        return 1
    if x == 'C':
        return 2
    return 3

def jyokencd(x):
    if x == '999':
        return 0
    if x == '016':
        return 1
    if x == '010':
        return 2
    if x == '005':
        return 3
    if x == '703':
        return 5
    if x == '702':
        return 5
    if x == '701':
        return 6
    return 4

def load():
    dbparams = "host={} user={} dbname={} port={} password={}".format(
            os.environ['host'],
            os.environ['user'],
            os.environ['dbname'],
            os.environ['port'],
            os.environ['password'],
        )
    
    query = """
    SELECT race.year, race.monthday, race.jyocd, race.racenum, race.kyori, race.trackcd,
    race.jyokencd5, race.gradecd, uma.harontimel3, uma.timediff, uma.futan,
    uma.kettonum, uma.bamei, uma.kakuteijyuni, uma.odds, uma.kisyucode,
    ketto.Ketto3InfoHansyokuNum1, ketto.Ketto3InfoHansyokuNum5
    FROM
    n_uma_race AS uma
    INNER JOIN n_race AS race
    ON uma.year = race.year
    AND uma.monthday = race.monthday
    AND uma.jyocd = race.jyocd
    AND uma.racenum = race.racenum
    INNER JOIN n_uma AS ketto
    ON uma.kettonum = ketto.kettonum
    WHERE uma.year::int > 2015
    AND race.jyocd in ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10');
    """

    PATH = "2015_2017_race.csv"
    if os.path.exists(PATH):
        df = pd.read_csv(PATH, dtype=str)
        df['racerank'] = df['jyokencd5'].map(jyokencd) + df['gradecd'].map(gradecd)
    else:
        with psycopg2.connect(dbparams) as conn:
            df = pd.io.sql.read_sql_query(query, conn)
            df.to_csv(PATH, index=False)
            df['racerank'] = df['jyokencd5'].map(jyokencd) + df['gradecd'].map(gradecd)
    df = df.sort_values(['year', 'monthday', 'jyocd', 'racenum', 'racerank'])
    # なしおよび00は19として扱う
    # df['bf_jyuni'] = df.groupby('kettonum')['kakuteijyuni'].shift(1).fillna('19').replace('00', '19')
    # df['bf_rank'] = df.groupby('kettonum')['racerank'].shift(1).fillna(4)
    # df['rankup'] =(df['bf_rank'] < df['racerank']).astype(np.int32)
    # df = df.dropna()

    return df

def load_racename(year, monthday):
    dbparams = "host={} user={} dbname={} port={} password={}".format(
            os.environ['host'],
            os.environ['user'],
            os.environ['dbname'],
            os.environ['port'],
            os.environ['password'],
        )
    
    query = """
    SELECT year, monthday, jyocd::int, racenum::int, ryakusyo10
    FROM n_race
    WHERE year = '%s'
    AND monthday = '%s'
    AND ryakusyo10 <> '';
    """ % (year, monthday)

    with psycopg2.connect(dbparams) as conn:
        df = pd.io.sql.read_sql_query(query, conn)

    return df

def split(df):
    mask = (df['year'].astype(int) < 2018)
    return df.pipe(lambda df: df[mask]), df.pipe(lambda df: df[~mask])
    