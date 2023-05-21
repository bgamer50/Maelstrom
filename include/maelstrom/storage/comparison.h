#pragma once

namespace maelstrom {
    enum comparator {
        GREATER_THAN = 0, // >
        LESS_THAN = 1, // <
        EQUALS = 2, // =
        GREATER_THAN_OR_EQUAL = 3, // >=
        LESS_THAN_OR_EQUAL = 4, // <=
        NOT_EQUALS = 5, // !=
        BETWEEN = 6, // [left, right)
        IS_IN = 7, // is in (a, b, c, ...)
        IS_NOT_IN = 8, // is not in (a, b, c, ...)
        INSIDE = 9, // (left, right)
        OUTSIDE = 10, // < left & > right
    };
}