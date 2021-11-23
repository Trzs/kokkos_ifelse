#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>

#define LOOK_INTO(var) if (pixIdx == 0) \
        { printf("%d (%s): %.9E \n", __LINE__, #var, var); }

int main() {

  Kokkos::initialize();
  {
#ifdef KOKKOS_ENABLE_CUDA
    std::cout << "Using Cuda" << std::endl;
#endif
#ifdef KOKKOS_ENABLE_HIP
    std::cout << "Using HIP" << std::endl;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    std::cout << "Using OpenMP" << std::endl;
#endif

    const int fpixels = 1536;
    const int spixels = 1536;
    const int total_pixels = 2;//fpixels * spixels;

    double capture_fraction = 1.0;
    float hrad_sqr = 2.0;
    double normal_name = 1.0;

    Kokkos::parallel_for("kokkosSpotsKernel", total_pixels, KOKKOS_LAMBDA(const int& pixIdx)
    {
        double captur_efraction = 1.0;
	double useful_name = 1.0;
        LOOK_INTO(capture_fraction)
	LOOK_INTO(hrad_sqr)
	LOOK_INTO(normal_name);
        LOOK_INTO(captur_efraction)
	LOOK_INTO(useful_name)

    });


  }
  Kokkos::finalize();

  return 0;
}
