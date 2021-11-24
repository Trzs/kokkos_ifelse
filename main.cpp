#include <iostream>
#include <Kokkos_Core.hpp>

#define LOOK_INTO(var) { printf("%d (%s): %.9E \n", __LINE__, #var, var); }

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

    double capture_fraction = 1.1;
    float hrad_sqr = 1.01;
    double normal_name = 1.001;

    printf("Inside main():\n");
    LOOK_INTO(capture_fraction)
    LOOK_INTO(hrad_sqr)
    LOOK_INTO(normal_name)

    Kokkos::parallel_for("kokkosSpotsKernel", 1, KOKKOS_LAMBDA(const int& pixIdx)
    {
    	double I = 0;
      double captur_efraction = 1.0001;
      double useful_name = 1.00001;
      printf("\nInside parallel_for():\n");
      LOOK_INTO(capture_fraction)
      LOOK_INTO(hrad_sqr)
      LOOK_INTO(normal_name);
      LOOK_INTO(captur_efraction)
      LOOK_INTO(useful_name)
  
      I = capture_fraction + hrad_sqr + normal_name + captur_efraction + useful_name;
      LOOK_INTO(I)
    });

  }
  Kokkos::finalize();

  return 0;
}
