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
      const int fpixel = pixIdx % fpixels;
      const int spixel = pixIdx / fpixels;

      CUDAREAL I = 0;

      int subF = 0;
      int subS = 0;
      int oversample = 2;

      CUDAREAL Fdet = subpixel_size * (fpixel * oversample + subF) + subpixel_size / 2.0;
      CUDAREAL Sdet = subpixel_size * (spixel * oversample + subS) + subpixel_size / 2.0;

      int thick_tic;
      for (thick_tic = 0; thick_tic < detector_thicksteps; ++thick_tic)
      {
        CUDAREAL Odet = thick_tic * detector_thickstep;
        LOOK_INTO(Odet)

        CUDAREAL pixel_pos[4];
        pixel_pos[1] = Fdet * 1 + Sdet * 0 + Odet * 0;
        pixel_pos[2] = Fdet * 0 + Sdet * 1 + Odet * 0;
        pixel_pos[3] = Fdet * 0 + Sdet * 0 + Odet * 1;

        CUDAREAL diffracted[4];
        CUDAREAL airpath = unitize(pixel_pos, diffracted); LOOK_INTO(airpath);

        // now calculate detector thickness effects
        CUDAREAL capture_fraction = 1.0;
        LOOK_INTO_INT(thick_tic)
        LOOK_INTO(detector_thickstep)
        LOOK_INTO(detector_thick)
        LOOK_INTO(detector_mu)
        LOOK_INTO(capture_fraction)
        if (detector_thick > 0.0 && detector_mu> 0.0) {
          // inverse of effective thickness increase
          CUDAREAL odet[4];
          odet[1] = odet_vector(1);
          odet[2] = odet_vector(2);
          odet[3] = odet_vector(3);
          CUDAREAL parallax = dot_product(odet, diffracted);
          LOOK_INTO(parallax)
          capture_fraction = exp(-thick_tic * detector_thickstep / detector_mu / parallax)
                           - exp(-(thick_tic + 1) * detector_thickstep / detector_mu / parallax);
        }
        LOOK_INTO(capture_fraction)

        I += capture_fraction;
      }

      result(pixIdx) = I;

    });


  }
  Kokkos::finalize();

  return 0;
}
