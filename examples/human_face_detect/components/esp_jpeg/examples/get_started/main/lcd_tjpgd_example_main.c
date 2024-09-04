/*
 * SPDX-FileCopyrightText: 2022-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */

#include <stdio.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_lcd_panel_ops.h"
#include "esp_heap_caps.h"
#include "pretty_effect.h"
#include "bsp/esp-bsp.h"
#include "bsp/display.h"

// Using SPI2 in the example, as it also supports octal modes on some targets
#define LCD_HOST       SPI2_HOST
// To speed up transfers, every SPI transfer sends a bunch of lines. This define specifies how many.
// More means more memory use, but less overhead for setting up / finishing transfers. Make sure 240
// is dividable by this.
#define PARALLEL_LINES CONFIG_EXAMPLE_LCD_FLUSH_PARALLEL_LINES
// The number of frames to show before rotate the graph
#define ROTATE_FRAME   30

#if BSP_LCD_H_RES > BSP_LCD_V_RES
#define EXAMPLE_LCD_SWAP    0
#define EXAMPLE_LCD_H_RES   BSP_LCD_H_RES
#define EXAMPLE_LCD_V_RES   BSP_LCD_V_RES
#else
#define EXAMPLE_LCD_SWAP    1
#define EXAMPLE_LCD_H_RES   BSP_LCD_V_RES
#define EXAMPLE_LCD_V_RES   BSP_LCD_H_RES
#endif

// Simple routine to generate some patterns and send them to the LCD. Because the
// SPI driver handles transactions in the background, we can calculate the next line
// while the previous one is being sent.
static uint16_t *s_lines[2];
static void display_pretty_colors(esp_lcd_panel_handle_t panel_handle)
{
    int frame = 0;
    // Indexes of the line currently being sent to the LCD and the line we're calculating
    int sending_line = 0;
    int calc_line = 0;

    // After ROTATE_FRAME frames, the image will be rotated
    while (frame <= ROTATE_FRAME) {
        frame++;
        for (int y = 0; y < EXAMPLE_LCD_V_RES; y += PARALLEL_LINES) {
            // Calculate a line
            pretty_effect_calc_lines(s_lines[calc_line], y, frame, PARALLEL_LINES);
            sending_line = calc_line;
            calc_line = !calc_line;
            // Send the calculated data
            esp_lcd_panel_draw_bitmap(panel_handle, 0, y, 0 + EXAMPLE_LCD_H_RES, y + PARALLEL_LINES, s_lines[sending_line]);
        }
    }
}

void app_main(void)
{
    esp_lcd_panel_io_handle_t io_handle = NULL;
    esp_lcd_panel_handle_t panel_handle = NULL;

    bsp_display_config_t disp_cfg = {
        .max_transfer_sz = EXAMPLE_LCD_H_RES * PARALLEL_LINES * sizeof(uint16_t),
    };
    // Display initialize from BSP
    bsp_display_new(&disp_cfg, &panel_handle, &io_handle);
    esp_lcd_panel_disp_on_off(panel_handle, true);
    bsp_display_backlight_on();

    // Initialize the effect displayed
    ESP_ERROR_CHECK(pretty_effect_init());

    // "Rotate or not" flag
    bool is_rotated = false;

    // Allocate memory for the pixel buffers
    for (int i = 0; i < 2; i++) {
        s_lines[i] = heap_caps_malloc(EXAMPLE_LCD_H_RES * PARALLEL_LINES * sizeof(uint16_t), MALLOC_CAP_DMA);
        assert(s_lines[i] != NULL);
    }

#if EXAMPLE_LCD_SWAP
    esp_lcd_panel_swap_xy(panel_handle, true);
#endif

    // Start and rotate
    while (1) {
        // Set driver configuration to rotate 180 degrees each time
        ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel_handle, is_rotated, is_rotated));
        // Display
        display_pretty_colors(panel_handle);
        is_rotated = !is_rotated;
    }
}
