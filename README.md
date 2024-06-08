# Traffic Flow Monitoring and Prediction using Artificial Intelligence

## Introduction

Welcome to the Traffic Flow Monitoring and Prediction using Artificial Intelligence project! This repository contains the code and resources necessary to monitor traffic flow by counting vehicles of different categories such as bikes, cars, trucks, and buses. The project aims to provide a comprehensive solution for real-time traffic monitoring and prediction using AI techniques.

## Features

- **Real-time Vehicle Counting**: The system continuously monitors traffic and counts vehicles in real-time, categorizing them into different types.
- **Alarm Triggering**: If the count of vehicles exceeds a certain threshold, an alarm is triggered to alert the control room.
- **Data Logging**: Each vehicle count is stored in an Excel sheet for further analysis and monitoring.
- **Traffic Prediction**: Leveraging the collected data, the system can predict future traffic conditions based on historical patterns.

## Screenshots
![image](https://github.com/umangmtiwari/Traffic-Flow-Monitoring-and-Prediction-using-Artificial-Intelligence/assets/143312015/7863d8e4-b6a6-4354-a189-fbd2abf4e3da)

![image](https://github.com/umangmtiwari/Traffic-Flow-Monitoring-and-Prediction-using-Artificial-Intelligence/assets/143312015/ddfe739c-234c-4cce-8d60-aaa4804c5fab)

![image](https://github.com/umangmtiwari/Traffic-Flow-Monitoring-and-Prediction-using-Artificial-Intelligence/assets/143312015/b2ea7338-0424-4683-b5c7-cbd10ebc26f7)

![image](https://github.com/umangmtiwari/Traffic-Flow-Monitoring-and-Prediction-using-Artificial-Intelligence/assets/143312015/b0e06a75-12f8-4925-9df7-6c01d492c93d)

## Usage

To use this project, follow these steps:

1. **Setup Environment**: Ensure you have Python installed along with necessary libraries specified in `requirements.txt`.
   
2. **Run the Code**: Execute the main script `traffic_monitor.py` to start monitoring traffic flow.
   
3. **Alarm Configuration**: Adjust the threshold for vehicle count in `config.yaml` to set when the alarm should trigger.
   
4. **Data Analysis**: Analyze the stored data in the Excel sheet to gain insights into traffic patterns and trends.

## Dependencies

This project relies on the following libraries:

- numpy
- pandas
- flask
- opencv-python
- datetime
- sklearn

Install them using pip:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions to this project are welcome! If you have any ideas for improvement, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

I would like to thank the open-source community for their valuable contributions and resources that made this project possible.
