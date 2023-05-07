#pragma once

#include "containers/vector.h"

namespace maelstrom {

    /*
        Removes elements that are "true" in stencil.  In other words,
        stencil is a boolean mask used to determine which elements to
        remove from the array.
    */
    void remove_if(maelstrom::vector& array, maelstrom::vector& stencil);

}