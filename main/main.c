#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"
#include "esp_spi_flash.h"
#include "esp_partition.h"
#include "sdkconfig.h"


void printTask(void *arg)
{
    #define BUF_SIZE 5 * 1024
    char *tasklist = calloc(1, BUF_SIZE);
    while (1) {
#if CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS
            memset(tasklist, 0 , BUF_SIZE);
            vTaskGetRunTimeStats(tasklist);
            printf("Running tasks CPU usage: \n %s\r\n", tasklist);
            printf("RAM size: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT));
            //ESP_LOGI(APP_TAG, "Running tasks CPU usage: \n %s\r\n", tasklist);
#endif
            vTaskDelay(17000 / portTICK_RATE_MS);
        }
    free(tasklist);
}


void app_main()
{
    printf("Start free RAM size: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
}
