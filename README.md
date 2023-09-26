# UOW-Agriculture-Threats-Detection
Detecting weeds and plant disease using state of the art computer vision models


## How to Run with Docker

### 1 - Docker Requirements

Before you build Docker Image, make sure you have enough space in your system because the docker image will take 5GB of your storage.

- Storage - 5GB
- RAM - 4GB/8GB

The docker desktop version I tested is 4.0.1 (68347). However, you should be able to build image with latest docker destop too.

### 2 - Builing Docker Image

- go to the root dir (**UOW-AGRICULTURE-THREATS-DETECTION**)
- open docker desktop 
- run below command in your terminal to build docker image

```bash
docker compose build
```

After succesful build, you should be seeing something similar as shown below.

<img src="assets/sample-1.png" width="800" height="400"><br>

### 3 - Running Docker Container

To run your container, run below command

```bash
docker compose up
```

Open one of the IP addresses as shown in your terminal and it will direct you to Flask Web Application.

<img src="assets/sample-2.png" width="850" height="400"><br>