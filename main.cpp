#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>

#define CUDAREAL double

#define LOOK_INTO(var) if (pixIdx == 297053) \
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
    const int total_pixels = fpixels * spixels;

    int detector_thicksteps = 1;
    CUDAREAL detector_thickstep = 0.0;
    CUDAREAL detector_thick = 0.0;
    CUDAREAL detector_mu = 2.34E-4;

    auto result = Kokkos::View<CUDAREAL[total_pixels]>("result");

    Kokkos::parallel_for("kokkosSpotsKernel", total_pixels, KOKKOS_LAMBDA(const int& pixIdx)
    {
      CUDAREAL I = 0;

      int thick_tic;
      //for (thick_tic = 0; thick_tic < detector_thicksteps; ++thick_tic)
      {

        CUDAREAL captur_efraction = 1.0;
        LOOK_INTO(detector_thick)
        LOOK_INTO(detector_mu)
        LOOK_INTO(captur_efraction)
        LOOK_INTO(captur_efraction)

        I += captur_efraction;
      }

      result(pixIdx) = I; LOOK_INTO(I)

    });


  }
  Kokkos::finalize();

  return 0;
}
