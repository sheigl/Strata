// Feasibility test for the per-stage mirror design: can a later split stage's GPU read host-pinned
// memory allocated under a DIFFERENT card's context? On SYCL this is host USM (sycl::malloc_host),
// which should be visible to all devices; the mirror uses a pinned allocation the device reads over
// PCIe. This test: allocate host USM, fill it, have device 1 (the B70) read it into a device buffer
// and back, and confirm the bytes match. If device 1 cannot read it (fault / zeros / wrong data),
// the per-stage-mirror design needs the alternative (per-stage pinned pools) path.
//
// Build (in the strata-sycl-dev image or a host with oneAPI):
//   icpx -fsycl mirror_xcard.cpp -o mirror_xcard
// Run with two cards visible:  ONEAPI_DEVICE_SELECTOR=level_zero:* ./mirror_xcard
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>

int main() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.size() < 2) { std::printf("need 2 GPUs, found %zu\n", devs.size()); return 2; }
    for (size_t i = 0; i < devs.size(); ++i)
        std::printf("dev %zu: %s\n", i, devs[i].get_info<sycl::info::device::name>().c_str());

    const size_t n = 1 << 20;                 // 4 MB of floats
    // Host USM allocated on a queue bound to dev 0 (the "first stage's" context).
    sycl::queue q0(devs[0]);
    float* host = sycl::malloc_host<float>(n, q0);
    if (!host) { std::printf("malloc_host failed\n"); return 1; }
    for (size_t i = 0; i < n; ++i) host[i] = (float) (i % 1000);

    // Device 1 (the "later stage") reads the host pointer directly in a kernel.
    sycl::queue q1(devs[1]);
    float* out = sycl::malloc_device<float>(n, q1);
    bool ok = true;
    try {
        q1.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { out[i] = host[i]; }).wait();
    } catch (const sycl::exception& e) {
        std::printf("dev1 kernel reading dev0-context host USM threw: %s\n", e.what());
        ok = false;
    }
    if (ok) {
        std::vector<float> back(n);
        q1.memcpy(back.data(), out, n * sizeof(float)).wait();
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) if (back[i] != (float) (i % 1000)) ++bad;
        std::printf("dev1 read of dev0-context host USM: %s (%zu/%zu wrong)\n",
                    bad == 0 ? "OK" : "MISMATCH", bad, n);
        ok = bad == 0;
    }
    sycl::free(out, q1);
    sycl::free(host, q0);
    std::printf("RESULT: %s\n", ok ? "cross-card host-mirror read WORKS" : "cross-card read FAILED - use per-stage pools");
    return ok ? 0 : 1;
}
