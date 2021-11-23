#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>

#define CUDAREAL double

#define LOOK_INTO(var) if (pixIdx == 297053) \
        { printf("%d (%s): %.9E \n", __LINE__, #var, var); }

#define LOOK_INTO_INT(var) if (pixIdx == 297053) \
        { printf("%d (%s): %d \n", __LINE__, #var, var); }

// vector inner product where vector magnitude is 0th element
KOKKOS_FUNCTION CUDAREAL dot_product(const CUDAREAL *x, const CUDAREAL *y) {
        return x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

// make provided vector a unit vector
KOKKOS_FUNCTION CUDAREAL unitize(CUDAREAL * vector, CUDAREAL * new_unit_vector) {

        CUDAREAL v1 = vector[1];
        CUDAREAL v2 = vector[2];
        CUDAREAL v3 = vector[3];

#ifdef __CUDA_ARCH__
        CUDAREAL mag = norm3d(v1, v2, v3);
#else
        CUDAREAL mag = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
#endif

        if (mag != 0.0) {
                new_unit_vector[0] = mag;
                new_unit_vector[1] = v1 / mag;
                new_unit_vector[2] = v2 / mag;
                new_unit_vector[3] = v3 / mag;
        } else {
                new_unit_vector[0] = 0.0;
                new_unit_vector[1] = 0.0;
                new_unit_vector[2] = 0.0;
                new_unit_vector[3] = 0.0;
        }
        return mag;
}

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

    const int N = 10;
    using view_t = Kokkos::View<int[N]>;

    auto odet_vector = Kokkos::View<CUDAREAL[4]>("odet");
    Kokkos::parallel_for("init", 1, KOKKOS_LAMBDA(const int& ind)
    {
      odet_vector(1) = 0.0;
      odet_vector(2) = 0.0;
      odet_vector(3) = 1.0;
    });

    const int fpixels = 1536;
    const int spixels = 1536;
    const int total_pixels = fpixels * spixels;
    CUDAREAL subpixel_size = 2.5E-5;

    int detector_thicksteps = 1;
    CUDAREAL detector_thickstep = 0.0;
    CUDAREAL detector_thick = 0.0;
    CUDAREAL detector_mu = 2.34E-4;

    auto result = Kokkos::View<CUDAREAL[total_pixels]>("result");

    Kokkos::parallel_for("kokkosSpotsKernel", total_pixels, KOKKOS_LAMBDA(const int& pixIdx)
    {
      CUDAREAL I = 0;

      int thick_tic;
      for (thick_tic = 0; thick_tic < detector_thicksteps; ++thick_tic)
      {

        // now calculate detector thickness effects
        CUDAREAL capture_fraction = 1.0;
        LOOK_INTO(detector_thick)
        LOOK_INTO(detector_mu)
        LOOK_INTO(capture_fraction)
        if (detector_thick > 0.0 && detector_mu> 0.0) {
	  capture_fraction = 2;
	  LOOK_INTO(capture_fraction)
        }
        LOOK_INTO(capture_fraction)

        I += capture_fraction;
      }

      result(pixIdx) = I; LOOK_INTO(I)

    });


  }
  Kokkos::finalize();

  return 0;
}
