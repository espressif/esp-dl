#include "dl_image_bmp.hpp"

static const char *TAG = "dl_image_bmp";
namespace dl {
namespace image {
#pragma pack(push, 1)
typedef struct {
    uint16_t file_type{0x4D42}; // File type always BM which is 0x4D42 (stored as hex uint16_t in little endian)
    uint32_t file_size{0};      // Size of the file (in bytes)
    uint16_t reserved1{0};      // Reserved, always 0
    uint16_t reserved2{0};      // Reserved, always 0
    uint32_t offset_data{0};    // Start position of pixel data (bytes from the beginning of the file)
} bmp_file_header_t;

typedef struct {
    uint32_t size{40};       // Size of this header (in bytes)
    int32_t width{0};        // width of bitmap in pixels
    int32_t height{0};       // width of bitmap in pixels
                             //       (if positive, bottom-up, with origin in lower left corner)
                             //       (if negative, top-down, with origin in upper left corner)
    uint16_t planes{1};      // No. of planes for the target device, this is always 1
    uint16_t bit_count{0};   // No. of bits per pixel
    uint32_t compression{0}; // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
    uint32_t size_image{0};  // 0 - for uncompressed images
    int32_t x_pixels_per_meter{0};
    int32_t y_pixels_per_meter{0};
    uint32_t colors_used{
        0}; // No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count
    uint32_t colors_important{0}; // No. of colors used for displaying the bitmap. If 0 all colors are required
} bmp_info_header_t;
#pragma pack(pop)

esp_err_t write_bmp_base(const img_t &img, const char *file_name)
{
    bmp_file_header_t bmp_file_header;
    bmp_info_header_t bmp_info_header;
    FILE *f = fopen(file_name, "wb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open %s.", file_name);
        return ESP_FAIL;
    }
    size_t size;

    int channel = (img.pix_type == DL_IMAGE_PIX_TYPE_GRAY) ? 1 : 3;
    int row_stride = img.width * channel;
    int row_stride_padded = DL_IMAGE_ALIGN_UP(row_stride, 4); // every row 4 byte align

    if (img.pix_type == DL_IMAGE_PIX_TYPE_GRAY) {
        uint8_t color_table[256 * 4];
        for (int i = 0; i < 256; i++) {
            color_table[4 * i] = i;
            color_table[4 * i + 1] = i;
            color_table[4 * i + 2] = i;
            color_table[4 * i + 3] = 0;
        }
        bmp_info_header.bit_count = 8;
        bmp_info_header.height = img.height;
        bmp_info_header.width = img.width;
        bmp_info_header.colors_used = 256;
        bmp_file_header.offset_data = sizeof(bmp_file_header_t) + sizeof(bmp_info_header_t) + sizeof(color_table);
        bmp_file_header.file_size = bmp_file_header.offset_data + row_stride_padded * img.height;
        size = fwrite(&bmp_file_header, sizeof(bmp_file_header_t), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to write bmp file header.");
            fclose(f);
            return ESP_FAIL;
        }
        size = fwrite(&bmp_info_header, sizeof(bmp_info_header_t), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to write bmp info header.");
            fclose(f);
            return ESP_FAIL;
        }
        size = fwrite(color_table, sizeof(color_table), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to write color table.");
            fclose(f);
            return ESP_FAIL;
        }
    } else if (img.pix_type == DL_IMAGE_PIX_TYPE_RGB888) {
        bmp_info_header.bit_count = 24;
        bmp_info_header.height = img.height;
        bmp_info_header.width = img.width;
        bmp_file_header.offset_data = sizeof(bmp_file_header_t) + sizeof(bmp_info_header_t);
        bmp_file_header.file_size = bmp_file_header.offset_data + row_stride_padded * img.height;
        size = fwrite(&bmp_file_header, sizeof(bmp_file_header_t), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to write bmp file header.");
            fclose(f);
            return ESP_FAIL;
        }
        size = fwrite(&bmp_info_header, sizeof(bmp_info_header_t), 1, f);
        if (size != 1) {
            ESP_LOGE(TAG, "Failed to write bmp info header.");
            fclose(f);
            return ESP_FAIL;
        }
    } else {
        ESP_LOGE(TAG, "Unsupported image pix type for bmp.");
        return ESP_FAIL;
    }

    // write img data
    uint8_t *img_ptr = (uint8_t *)img.data + (img.height - 1) * row_stride;
    if (row_stride_padded == row_stride) {
        for (int i = 0; i < img.height; i++) {
            size = fwrite(img_ptr, row_stride, 1, f);
            if (size != 1) {
                ESP_LOGE(TAG, "Failed to write image data.");
                fclose(f);
                return ESP_FAIL;
            }
            img_ptr -= row_stride;
        }
    } else {
        uint8_t padding[row_stride_padded - row_stride]{};
        for (int i = 0; i < img.height; i++) {
            size = fwrite(img_ptr, row_stride, 1, f);
            if (size != 1) {
                ESP_LOGE(TAG, "Failed to write image data.");
                fclose(f);
                return ESP_FAIL;
            }
            size = fwrite(padding, row_stride_padded - row_stride, 1, f);
            if (size != 1) {
                ESP_LOGE(TAG, "Failed to write image data padding.");
                fclose(f);
                return ESP_FAIL;
            }
            img_ptr -= row_stride;
        }
    }
    fclose(f);
    return ESP_OK;
}

esp_err_t write_bmp(const img_t &img, const char *file_name, uint32_t caps)
{
    if (img.pix_type != DL_IMAGE_PIX_TYPE_RGB565 && img.pix_type != DL_IMAGE_PIX_TYPE_RGB888 &&
        img.pix_type != DL_IMAGE_PIX_TYPE_GRAY) {
        ESP_LOGE(TAG, "Unsupported img type.");
        return ESP_FAIL;
    }
    img_t img2write = img;
    bool free = false;
    if (img.pix_type == DL_IMAGE_PIX_TYPE_RGB565 ||
        (img.pix_type == DL_IMAGE_PIX_TYPE_RGB888 && (caps & DL_IMAGE_CAP_RGB_SWAP))) {
        img2write.pix_type = DL_IMAGE_PIX_TYPE_RGB888;
        img2write.data = heap_caps_malloc(get_img_byte_size(img2write), MALLOC_CAP_DEFAULT);
        convert_img(img, img2write, caps);
        free = true;
    };
    esp_err_t ret = write_bmp_base(img2write, file_name);
    if (free) {
        heap_caps_free(img2write.data);
    }
    return ret;
}

img_t read_bmp(const char *file_name)
{
    img_t img;
    bmp_file_header_t bmp_file_header;
    bmp_info_header_t bmp_info_header;
    FILE *f = fopen(file_name, "rb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open %s.", file_name);
        return {};
    }
    fread(&bmp_file_header, sizeof(bmp_file_header_t), 1, f);
    fread(&bmp_info_header, sizeof(bmp_info_header_t), 1, f);

    // check file type
    if (bmp_file_header.file_type != 0x4D42) {
        ESP_LOGE(TAG, "It is not a bmp file.");
        fclose(f);
        return {};
    }

    // set img info
    if (bmp_info_header.bit_count == 8) {
        img.pix_type = DL_IMAGE_PIX_TYPE_GRAY;
    } else if (bmp_info_header.bit_count == 24) {
        img.pix_type = DL_IMAGE_PIX_TYPE_RGB888;
    }
    img.height = bmp_info_header.height;
    img.width = bmp_info_header.width;
    int channel = (img.pix_type == DL_IMAGE_PIX_TYPE_GRAY) ? 1 : 3;
    img.data = heap_caps_malloc(img.height * img.width * channel, MALLOC_CAP_DEFAULT);
    if (!img.data) {
        fclose(f);
        ESP_LOGE(TAG, "Failed to alloc memory");
        return {};
    }
    int row_stride = img.width * channel;
    int row_stride_padded = DL_IMAGE_ALIGN_UP(row_stride, 4); // every row 4 byte align

    if (fseek(f, 0, SEEK_END) != 0) {
        ESP_LOGE(TAG, "Failed to seek bmp file.");
        fclose(f);
        return {};
    }
    long pos = ftell(f);
    // check file size
    if (bmp_file_header.file_size != pos) {
        ESP_LOGE(TAG, "Bmp file size does not match bmp info header.");
        fclose(f);
        return {};
    }
    // check img data size
    if (pos - bmp_file_header.offset_data != img.height * row_stride_padded) {
        ESP_LOGE(TAG, "Image data size does not match bmp info header.");
        fclose(f);
        return {};
    }

    // read img data
    uint8_t *img_ptr = (uint8_t *)img.data + (img.height - 1) * row_stride;
    if (fseek(f, bmp_file_header.offset_data, SEEK_SET) != 0) {
        ESP_LOGE(TAG, "Failed to seek bmp file.");
        fclose(f);
        return {};
    }

    for (int i = 0; i < img.height; i++) {
        fread(img_ptr, 1, row_stride, f);
        if (fseek(f, row_stride_padded - row_stride, SEEK_CUR) != 0) {
            ESP_LOGE(TAG, "Failed to seek bmp file.");
            fclose(f);
            return {};
        }
        img_ptr -= row_stride;
    }
    fclose(f);
    return img;
}
} // namespace image
} // namespace dl
