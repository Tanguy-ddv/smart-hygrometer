"""
This file contains the first exercise of the homework 1 of the course
'Machine Learning for IoT'
2024 - 2025
Group 4
"""
# =========================
# Redis writer
# ========================
from dataclasses import dataclass
import time
from redis import Redis

@dataclass
class RedisCredential:
    """
    The RedisCredential is a class used to store users data.
    """
    username: str
    password: str
    host: str
    port: str

TanguyRedis = RedisCredential(
    "default",
    "pKYsQDqLOuiMwEMh9T00bNH3K50UNlYG",
    "redis-17471.c339.eu-west-3-1.ec2.redns.redis-cloud.com",
    17471
)

class RedisWriter:
    """
    The RedisWriter class is used to manage the redis database.
    """

    def __init__(
        self,
        credentials: RedisCredential,
        timeseries: list[str],
        chunk_sizes_bytes: list[int] | int = 4096,
        retentions_hour: list[int] | int = 24,
        compresses: list[bool] | bool = True,
        delete_if_exists: bool = False
    ):
        # Create the redis client
        self.redis_client = Redis(
            host=credentials.host,
            port=credentials.port,
            username=credentials.username,
            password=credentials.password
        )

        # Refactor the timeseries arguments
        if isinstance(chunk_sizes_bytes, int):
            chunk_sizes_bytes = [chunk_sizes_bytes]*len(timeseries)
        if isinstance(retentions_hour, int):
            retentions_hour = [retentions_hour]*len(timeseries)
        if isinstance(compresses, bool):
            compresses = [compresses]*len(timeseries)
        
        # Delete the previous measures if needed.
        if delete_if_exists:
            for ts in timeseries:
                try:
                    self.redis_client.delete(ts)
                except:
                    pass

        # Create the timeseries
        for ts, csb, rh, c in zip(timeseries, chunk_sizes_bytes, retentions_hour, compresses):
            try:
                self.redis_client.ts().create(
                    ts,
                    retention_msecs=rh*60*60*1000,
                    chunk_size=csb,
                    uncompressed = not c
                )
            except:
                pass
        
        # Verify the connection
        is_connected = self.redis_client.ping()
        print("Successfully connected:", is_connected)
    
    def add(self, ts, value):
        """Add a new value to the time series."""
        timestamp = int(time.time()*1000)
        self.redis_client.ts().add(ts, timestamp, value)
    
    def info(self, ts):
        """Return the info object of the time series."""
        return self.redis_client.ts().info(ts)

    def get_keys(self):
        """return all the keys of the time series."""
        return self.redis_client.keys('*')

    def create_rule(self, src: str, dest: str, kind: str, duration_h: int):
        self.redis_client.ts().createrule(src, dest, kind, duration_h*60*60*1000)

# ==========================
#  Humidity temperature sensor reader
# ==========================
import adafruit_dht
import uuid
from board import D4

class HTReader:
    """The HT Reader os a wrapper to adafruit_dht11 used to get measure the temperature and the humidity from the sensor."""

    def __init__(self) -> None:
        self.dht = adafruit_dht.DHT11(D4)
        self.mac_adress = hex(uuid.getnode())

    def humidity(self):
        try:
            humidity = self.dht.humidity
        except:
            humidity = None
            self.dht.exit()
            self.dht = adafruit_dht.DHT11(D4)

        return humidity

    def temp(self):
        try:
            temp = self.dht.temperature
        except:
            temp = None
            self.dht.exit()
            self.dht = adafruit_dht.DHT11(D4)
        return temp

# =========================
# Command line inputs
# ========================

import argparse
def parse_credentials():
    """Parse the credentials from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-h", "--host", default=None)
    parser.add_argument("-r", "--port", default=None)
    parser.add_argument("-u", "--user", default='default')
    parser.add_argument("-p", "--password", default=None)
    args = parser.parse_args()
    if args.host is None:
        raise ValueError("Host can't be None, please specify an host with --host")
    if args.port is None:
        raise ValueError("Port can't be None, please specify a port with --port")
    if args.user is None:
        print("No user was mentioned, default value used is 'default'. You can change it by specifying --user")
    if args.password is None:
        raise ValueError("Password can't be None, please specify a password with --password")
    return RedisCredential(args.user, args.password, args.host, args.port)

# ========================
# Create the database

def create_rdwriter(credentials: RedisCredential, mac_address: str):
    timeseries = [
        mac_address + ':temperature',
        mac_address + ':humidty',
        mac_address + ':humidity_min',
        mac_address + ':humidity_max',
        mac_address + ':humidty_min',
        mac_address + ':temperature_min',
        mac_address + ':temperature_max',
        mac_address + ':temperature_min'
    ] # All the time series created
    rentention_periods = [30*24, 30*24, 365*24, 365*24, 365*24, 365*24, 365*24, 365*24]
    # All the retention periods in hour, retention_period[i] refers to timeseries[i]
    rdwriter = RedisWriter(
        credentials,
        timeseries,
        retentions_hour=rentention_periods,
        compresses=True,
        delete_if_exists=False
        )
    
    return rdwriter, tuple(timeseries)

def main(parse_credential: bool):
    """
    Main script to be executed.
    If parse_credential is false, use Tanguy's database.
    """

    if parse_credential:
        credentials = parse_credentials()
    else:
        credentials = TanguyRedis
    ht_reader = HTReader()
    # Create the redis writer and get the timeseries names
    rdwriter, (tts, hts, minhts, maxhts, avghts, mintts, maxtts, avgtts) = create_rdwriter(credentials, ht_reader.mac_adress)
    # Add rules for aggregation
    rdwriter.create_rule(tts, avgtts, 'avg', 24*30)
    rdwriter.create_rule(tts, mintts, 'min', 24*30)
    rdwriter.create_rule(tts, maxtts, 'max', 24*30)

    rdwriter.create_rule(hts, avghts, 'avg', 24*30)
    rdwriter.create_rule(hts, minhts, 'min', 24*30)
    rdwriter.create_rule(hts, maxhts, 'max', 24*30)

    while True:
        temperature = ht_reader.temp()
        humidity = ht_reader.humidity()
        # Add the measure to the db and the history
        if temperature != None:
            rdwriter.add(tts, temperature)
        if humidity != None:
            rdwriter.add(hts, humidity)
        
        time.sleep(2)

if __name__ == '__main__':
    parse_credential = True
    main(parse_credential)