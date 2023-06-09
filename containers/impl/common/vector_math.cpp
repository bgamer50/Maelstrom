#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/math.h"

namespace maelstrom {
    vector vector::operator+(vector& other) {
        return maelstrom::math(*this, other, PLUS);    
    }

    vector vector::operator-(vector& other) {
        return maelstrom::math(*this, other, MINUS);    
    }

    vector vector::operator*(vector& other) {
        return maelstrom::math(*this, other, TIMES);    
    }

    vector vector::operator/(vector& other) {
        return maelstrom::math(*this, other, DIVIDE);    
    }
}