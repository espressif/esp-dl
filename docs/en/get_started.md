# Get Started

This document tells how to develop with ESP-DL.

## Get ESP-IDF

ESP-DL runs based on ESP-IDF. See [https://idf.espressif.com/](https://idf.espressif.com/) for links to detailed instructions on how to set up the ESP-IDF depending on chip you use.



## Get ESP-DL and Run Example

1. Git clone or download ESP-DL.

    ```shell
    git clone https://github.com/espressif/esp-dl.git
    ```

2. Open a terminal. Get into the [tutorial](../../tutorial/) folder. Or, you can try other examples in [ESP-DL/examples](../../examples).

3. Set target chip. For example, if target chip is ESP32, then run

    ```shell
    idf.py set-target esp32
    ```

4. Flash and Monitor. Result printed in terminal.

    ```shell
    idf.py flash monitor
    ```
    
    - If target chip is ESP32, then
      
      ```shell
      MNIST::forward: 37294 us
      Prediction Result: 9
      ```

    
    - If target chip is ESP32-S3, then

      ```shell
      MNIST::forward: 6103 us
      Prediction Result: 9
      ```



## Become Component

ESP-DL is just a library contains various deep-learning API. We recommend to make ESP-DL as a component in a project. Take [ESP-WHO](https://github.com/espressif/esp-dl.git) for example, ESP-DL becomes a submodule in [ESP-WHO/components/](https://github.com/espressif/esp-who/tree/master/components).
