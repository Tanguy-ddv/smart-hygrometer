# Smart Hygrometer and thermometer

This repository contains the work of Tanguy Dugas du Villard, (https://github.com/Tanguy-ddv), Muhammad Nouman Siddiqui (https://github.com/nauman-siddiqui), and Shadi Mahboubpardah (https://github.com/shadimahboub) during the course "Machine Learning for IoT" at Politecnico di Torino, academic year 2024 - 2025.

The smart hygrometer has the following characteristics:

A Python script is running on a Raspberry Pi 4, model B with 2GB of memory, with a Linux OS. The pins of the computer are connected to a DHT sensor collecting the temperature and humidity of the surroundings. One USB port of the computer is connected to a microphone.
The script is continuously listening through the microphone and toggle detection when hearing "up" or "down". This keyword recognition is made by an optimized tflite model using a signal preprocessed into MFCC coefficients.
The collected data is then sent to an MQTT message broker, whose subscriber stores data in a Redis Database.

A rest server is also developed to retrieve the data requested in a given time range, and a rest client is developed to query it.
