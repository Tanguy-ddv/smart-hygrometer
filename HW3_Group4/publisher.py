import adafruit_dht
import uuid
import time
from board import D4
import paho.mqtt.client as mqtt
import json

# Create a new MQTT client
client = mqtt.Client()

# Connect to the MQTT broker
client.connect('mqtt.eclipseprojects.io', 1883)

mac_address = hex(uuid.getnode())
print(mac_address)
dht_device = adafruit_dht.DHT11(D4)

while True:
    timestamp = int(time.time()*1000)
    try:
        # Read the measures
        temperature = dht_device.temperature
        humidity = dht_device.humidity

        print(f'{timestamp} - {mac_address}:temperature = {temperature}')
        print(f'{timestamp} - {mac_address}:humidity = {humidity}')

        message = {
            'mac_address': mac_address,
            'timestamp': timestamp,
            'temperature': temperature,
            'humidity': humidity,
        }
        message_json = json.dumps(message)

        # Send the message.
        client.publish('s329112', message_json)

    except:
        # The device failed
        print(f'{timestamp} - sensor failure')
        dht_device.exit()
        dht_device = adafruit_dht.DHT11(D4)
    
    time.sleep(2)

