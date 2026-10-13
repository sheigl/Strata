// SYCL port: experts mirrored in pinned host memory (GgufExpertSource::mirror) in the device-built verify plan.
#pragma once
#include <cstdint>
namespace strata::kernels {
/// From now on (and in every graph captured after this call) the device-built plan (resident_plan) treats an expert
/// that is not in the VRAM cache but has a non-zero entry in `mirror_table` ([n_layers][n_expert] device-readable
/// addresses, 0 = none) as planned: the expert kernels read it from that address, over PCIe. `d_res` is the residency
/// table whose per-layer slices resident_plan receives, so the layer of a call is found from its pointer.
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table, int32_t* usage = nullptr);
/// Per-verifier variant of resident_plan (STRATA_SPLIT_MIRROR): the mirror table is passed explicitly instead of
/// read from the globals above, so each layer-split stage's verifier uses its own table (allocated on that stage's
/// device context) and concurrent stage verifies never touch shared state. `mirror_res` is the residency table the
/// table is indexed against (res_layer - mirror_res gives the layer's slot row); both null = no mirror.
void resident_plan_mirror(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                          const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                          long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err,
                          const int32_t* mirror_res, const unsigned long long* mirror_table);
}
