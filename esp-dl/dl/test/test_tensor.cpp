#include "unity.h"
#include <iostream>
#include <limits.h>

#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "dl_variable.hpp"

using namespace dl;
using namespace std;

TEST_CASE("Tensor", "shape")
{
    Tensor<int8_t> b;
    b.set_shape({2, 3, 4}).malloc_element();
    for (int i = 0; i < b.get_size(); i++) {
        b.element[i] = i;
    }

    Tensor<int8_t> c;
    c.set_shape({2, 3, 4}).malloc_element();
    for (int i = 0; i < b.get_size(); i++) {
        c.element[i] = i + 100;
    }

    Tensor<int8_t> x;
    x.set_shape({2, 3, 4, 3, 2}).malloc_element();
    for (int i = 0; i < x.get_size(); i++) {
        x.element[i] = i;
    }

    Tensor<int8_t> a(b, true);
    a.flatten();
    printf("\n---------------------------------------------------data--------------------------------------------------"
           "-\n");
    a.print({}, "\na orig");
    b.print({}, "\nb orig");
    c.print({}, "\nc orig");
    x.print({}, "\nc orig");

    printf("\n---------------------------------------------------slice-------------------------------------------------"
           "--\n");
    Tensor<int8_t> d = b.slice({0, -1, 1, 3, -1, 4});
    d.print({}, "\nslice 1");

    d = b.slice({0, 2, 0, 3, 0, 4});
    d.print({}, "\nslice 2");

    d = b.slice({0, 3, 2, 100, 0, 100});
    d.print({}, "\nslice 3");

    d = a.slice({-1, 100});
    d.print({}, "\nslice 4");
    d = a.slice({1, 20});
    d.print({}, "\nslice 5");

    a.expand_dims(1);
    d = a.slice({1, 20, 0, 1});
    d.print({}, "\nslice 6");
    a.expand_dims(0);
    d = a.slice({0, 1, 1, 20, 0, 1});
    d.print({}, "\nslice 7");

    a.squeeze();
    printf("\n---------------------------------------------------set value (T) "
           "---------------------------------------------------\n");
    d = b;
    d.set_value(-6);
    d.print({}, "\n set_value T 1");
    d.set_value(200);
    d.print({}, "\n set_value T 2");

    printf("\n---------------------------------------------------set value "
           "(Tensor<T>)---------------------------------------------------\n");
    Tensor<int8_t> temp;

    d.set_value(b);
    d.print({}, "\n set_value Tensor 1");

    temp = b.slice({0, 1, 0, 3, 0, 4});
    temp.print({}, "\n temp");
    d.set_value(temp);
    d.print({}, "set_value Tensor 2");

    temp = b.slice({0, 100, 0, 1, 0, 100});
    temp.print({}, "\n temp");
    d.set_value(temp);
    d.print({}, "\n set_value Tensor 3");

    temp = b.slice({0, 100, 0, 100, 0, 1});
    temp.print({}, "\n temp");
    d.set_value(temp);
    d.print({}, "set_value Tensor 4");

    d.flatten().set_value(a);
    d.print({}, "\nset_value Tensor 5");

    temp = a.slice({2, 3});
    d.set_value(temp);
    d.print({}, "\nset_value Tensor 6");

    temp = a.slice({2, 3}).expand_dims({0, 2});
    d.expand_dims({0, 2}).set_value(temp);
    d.print({}, "\nset_value Tensor 7");

    printf("\n---------------------------------------------------set value (slice, "
           "T)---------------------------------------------------\n");
    d.reshape({2, 3, -1}).set_value(b);
    d.set_value({0, 2, 1, 3, 0, 3}, 100);
    d.print({}, "\nset_value slice T 1");

    d.set_value(b);
    d.set_value({-1, 2, 0, 1, 3, 4}, 100);
    d.print({}, "\nset_value slice T 2");

    d.set_value(b);
    d.set_value({0, 100, 0, 100, 0, 100}, 100);
    d.print({}, "\nset_value slice T 3");

    d.flatten().set_value(a);
    d.set_value({0, -1}, -1);
    d.print({}, "\nset_value slice T 4");

    d.set_value({3, 99}, 66);
    d.print({}, "\nset_value slice T 5");

    printf("\n---------------------------------------------------set value (slice, "
           "Tensor<T>)---------------------------------------------------\n");

    d.reshape({2, 3, -1});
    temp = c.slice({0, 2, 1, 3, 0, -1});
    temp.print({}, "\n temp");
    d.set_value(b).set_value({0, 2, 1, 3, 0, 3}, temp);
    d.print({}, "set_value slice Tensor 1");

    temp = c.slice({0, 2, 1, 2, 0, 1});
    temp.print({}, "\n temp");
    d.set_value(b).set_value({0, 2, 1, 2, 0, 1}, temp);
    d.print({}, "set_value slice Tensor 2");

    temp = c.slice({0, 2, 1, 2, 0, 5});
    temp.print({}, "\n temp");
    d.set_value(b).set_value({0, 2, 1, 2, 0, 5}, temp);
    d.print({}, "set_value slice Tensor 3");

    temp = c.slice({0, 1, 1, 5, 0, 1});
    temp.print({}, "\n temp");
    d.set_value(b).set_value({0, 1, 1, 5, 0, 1}, temp);
    d.print({}, "set_value slice Tensor 4");

    temp = c.slice({0, 2, 1, 3, 0, 1});
    temp.print({}, "\n temp");
    d.set_value(b).set_value({0, 2, 1, 3, 0, 3}, temp);
    d.print({}, "set_value slice Tensor 5");

    temp = c.slice({0, 2, 1, 2, 0, 1});
    temp.print({}, "\n temp");
    d.set_value(b).set_value({0, 2, 1, 2, 0, 100}, temp);
    d.print({}, "set_value slice Tensor 6");

    temp = c.slice({0, 2, 1, 2, 0, 5});
    temp.print({}, "\n temp");
    d.set_value(b).set_value({0, 2, 1, 100, 0, 5}, temp);
    d.print({}, "set_value slice Tensor 7");

    temp = c.slice({0, 1, 1, 5, 0, 1});
    temp.print({}, "\n temp");
    d.set_value(b).set_value({0, 100, 1, 5, 0, 100}, temp);
    d.print({}, "set_value slice Tensor 8");

    d = x;
    temp = c.slice({0, 2, 1, 2, 0, 2}).expand_dims({0, 4});
    temp.print({}, "\n temp");
    d.set_value({0, 100, 0, 2, 1, 100, 0, 2, 0, 100}, temp);
    d.print({}, "set_value slice Tensor 9");

    printf("\n---------------------------------------------------reverse-----------------------------------------------"
           "----\n");

    d = b;
    d.reverse({0});
    d.print({}, "\nreverse {0}");

    d = b;
    d.reverse({1});
    d.print({}, "\nreverse {1}");

    d = b;
    d.reverse({2});
    d.print({}, "\nreverse {2}");

    d = b;
    d.reverse({-1});
    d.print({}, "\nreverse {-1}");

    d = b;
    d.reverse({-2});
    d.print({}, "\nreverse {-2}");

    d = b;
    d.reverse({1, 0});
    d.print({}, "\nreverse {1, 0}");

    d = b;
    d.reverse({1, 2});
    d.print({}, "\nreverse {1, 2}");

    d = b;
    d.reverse({2, 0});
    d.print({}, "\nreverse {2, 0}");

    d = b;
    d.reverse({2, 0, 1});
    d.print({}, "\nreverse {2, 0, 1}");

    d = b;
    d.reverse({});
    d.print({}, "\nreverse {}");

    d = a;
    d.reverse({-1});
    d.print({}, "\nreverse {-1}");
}
