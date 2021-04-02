#pragma once

#include "dl_define.hpp"
#include <stdio.h>
#include <vector>

namespace dl
{
    /**
     * @brief 
     * 
     * @tparam T 
     */
    template <typename T>
    class Feature
    {
    private:
        int size;       /*<! size of element */
        bool auto_free; /*<! free element when destroy */

    public:
        T *element;                          /*<! element of feature */
        int exponent;                        /*<! */
        std::vector<int> shape;              /*<! */
        std::vector<int> shape_with_padding; /*<! */
        std::vector<int> padding;            /*<! For 2D feature, padding format is [top, bottom, left, right] */

        /**
         * @brief Construct a new Feature object
         * 
         */
        Feature();

        /**
         * @brief Construct a new Feature object
         * 
         * @param feature 
         * @param padding 
         * @param auto_free 
         */
        Feature(Feature<T> &feature, std::vector<int> padding, const bool auto_free = true);

        Feature(Feature<T> &feature, bool deep);

        /**
         * @brief Destroy the Feature object
         * 
         */
        ~Feature();

        /**
         * @brief Set the auto free object
         * 
         * @param auto_free 
         * @return Feature<T>& 
         */
        Feature<T> &set_auto_free(const bool auto_free)
        {
            this->auto_free = auto_free;
            return *this;
        }

        /**
         * @brief Set the element object
         * 
         * @param element 
         * @return Feature& 
         */
        Feature<T> &set_element(T *element, const bool auto_free = false);

        /**
         * @brief Set the exponent object
         * 
         * @param exponent 
         * @return Feature& 
         */
        Feature &set_exponent(const int exponent);

        /**
         * @brief Set the shape object
         * 
         * @param shape 
         * @return Feature& 
         */
        Feature &set_shape(const std::vector<int> shape);

        /**
         * @brief Set the pad object
         * 
         * If this->element != NULL, free this->element, new an element with padding for this->element
         * 
         * @param padding 
         * @return Feature& 
         */
        Feature &set_padding(std::vector<int> &padding);

        /**
         * @brief Get the element object
         * 
         * @param padding 
         * @return T* 
         */
        T *get_element_ptr(const std::vector<int> padding = {0, 0, 0, 0})
        {
            return this->element + ((this->padding[0] - padding[0]) * this->shape_with_padding[1] + (this->padding[2] - padding[2])) * this->shape_with_padding[2];
        }

        T &get_element_value(const std::vector<int> index, const bool with_padding = false)
        {
            int i = 0;
            if (index.size() == 3)
            {
                int y = index[0];
                int x = index[1];
                int c = index[2];
                i = with_padding ? (y * this->shape_with_padding[1] + x) * this->shape_with_padding[2] + c : ((y + this->padding[0]) * this->shape_with_padding[1] + x + this->padding[2]) * this->shape_with_padding[2] + c;
            }
            else if (index.size() == 2)
            {
                // TODO: 1D
                printf("Not implement!\n");
            }
            else
            {
                printf("ERROR\n");
            }

            return this->element[i];
        }

        /**
         * @brief Get the size object
         * 
         * @return int 
         */
        int get_size();

        /**
         * @brief calloc element only if this->element == NULL
         * 
         */
        bool calloc_element(const bool auto_free = true);

        /**
         * @brief free element only if this->element != NULL
         * set this->element to NULL, after free
         */
        void free_element();

        void print_shape();

        /**
         * @brief 
         * 
         * @param y_start 
         * @param y_end 
         * @param x_start 
         * @param x_end 
         * @param c
         */
        void print2d(const int y_start, const int y_end, const int x_start, const int x_end, const int c, const char *message, const bool with_padding = false);

        bool check(T *gt_element, int bias = 2, bool info = true)
        {
            if (info)
                this->print_shape();
            int i = 0;
            for (int y = 0; y < this->shape[0]; y++)
            {
                for (int x = 0; x < this->shape[1]; x++)
                {
                    for (int c = 0; c < this->shape[2]; c++)
                    {
                        int a = this->get_element_value({y, x, c});
                        int b = gt_element[i];
                        int offset = DL_ABS(a - b);
                        if (offset > bias) // rounding mode is different between ESP32 and Python
                        {
                            printf("element[%d, %d, %d]: %d v.s. %d\n", y, x, c, a, b);
                            return false;
                        }
                        i++;
                    }
                }
            }

            if (info)
                printf("PASS\n");

            return true;
        }

        bool check_same_shape(Feature<T> &feature)
        {
            if (feature.shape.size() != this->shape.size())
            {
                return false;
            }
            for (int i = 0; i < this->shape.size(); i++)
            {
                if (feature.shape[i] != this->shape[i])
                {
                    return false;
                }
            }
            return true;
        }
    };
} // namespace dl