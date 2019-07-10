//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright 2018 Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_CUDA_CUDA_GRAPH_CUH
#define RIPPLES_CUDA_CUDA_GRAPH_CUH

#include <cassert>

#include "ripples/cuda/cuda_generate_rrr_sets.h"
#include "ripples/graph.h"

namespace ripples {

//! \brief Construct a device-side CUDA Graph from a host-side Graph.
//!
//! \param hg The host-side Graph to be mirrored.
cuda_device_graph *make_cuda_graph(const cuda_GraphTy &hg);

//! \brief Destroy a device-side CUDA Graph.
//!
//! \param hg The device-side CUDA Graph to be destroyed.
void destroy_cuda_graph(cuda_device_graph *g);

}  // namespace ripples

#endif  // RIPPLES_CUDA_CUDA_GRAPH_CUH
